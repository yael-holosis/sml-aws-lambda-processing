import boto3
import pymysql
import os
import io
import mne
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from sqlalchemy import create_engine
from typing import Optional, Dict, List, Any, Tuple, NamedTuple

class Config:
    SECRET_NAME = 'system_db/credentials'
    BUCKET_NAME = 'sleep-lab'
    REGION = 'us-east-1'
    EDF_SUFFIX = '.edf'
    VIDEO_SUFFIX = 'face_blur.mp4'
    HYPNO_SUFFIX = ''  # Empty since hypnogram files don't have a standard suffix
    REFERENCE_TYPE_ID = 5  # EDF
    EDF_FILES_PREFIX = 'edf_files/'

class SessionInfo(NamedTuple):
    """Holds information about a session extracted from folder name"""
    patient_name: str
    session_date: str
    session_id: Optional[int]
    folder_path: str

class AWSClients:
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=Config.REGION)
        self.secrets_client = boto3.client('secretsmanager', region_name=Config.REGION)

class S3Operations:
    def __init__(self, aws_clients: AWSClients):
        self.aws_clients = aws_clients

    def list_subfolders(self, bucket: str, prefix: str) -> List[str]:
        """List all subfolders under a prefix"""
        paginator = self.aws_clients.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')

        subfolders = []
        for page in page_iterator:
            for prefix in page.get('CommonPrefixes', []):
                subfolders.append(prefix['Prefix'])
        return subfolders

    def list_files_in_subfolder(self, bucket: str, subfolder: str, suffix: Optional[str] = None) -> List[str]:
        """List files in a specific subfolder with optional suffix filter"""
        paginator = self.aws_clients.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=subfolder)

        file_keys = []
        for page in page_iterator:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if suffix is None or key.endswith(suffix):
                    file_keys.append(key)
        return file_keys

    def list_files_without_extension(self, bucket: str, subfolder: str) -> List[str]:
        """List files in a specific subfolder that have no extension"""
        paginator = self.aws_clients.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=subfolder)

        file_keys = []
        for page in page_iterator:
            for obj in page.get('Contents', []):
                key = obj['Key']
                # Check if it's a direct file under the subfolder and has no extension
                if (key.count('/') == 2 and 
                    not key.endswith('/') and
                    '.' not in key.split('/')[-1]):
                    file_keys.append(key)
        return file_keys

    def get_unprocessed_files(self, bucket: str, subfolder: str, cursor: pymysql.cursors.DictCursor) -> Dict[str, List[str]]:
        """Get dictionary of unprocessed files by type"""
        # Get all files
        edf_keys = self.list_files_in_subfolder(bucket, subfolder, Config.EDF_SUFFIX)
        hypno_keys = self.list_files_without_extension(bucket, subfolder)
        video_keys = self.list_files_in_subfolder(bucket, subfolder, Config.VIDEO_SUFFIX)

        # Get processed files from DB
        cursor.execute("SELECT FilePath FROM SmlProcessedS3Files WHERE FilePath LIKE %s", (f"{subfolder}%",))
        processed_files = {row['FilePath'] for row in cursor.fetchall()}

        # Filter out processed files
        return {
            'edf': [k for k in edf_keys if k not in processed_files],
            'hypno': [k for k in hypno_keys if k not in processed_files],
            'video': [k for k in video_keys if k not in processed_files]
        }

class DatabaseConnection:
    def __init__(self, aws_clients: AWSClients):
        self.aws_clients = aws_clients

    def get_db_connection(self) -> pymysql.connections.Connection:
        secret = self.aws_clients.secrets_client.get_secret_value(SecretId=Config.SECRET_NAME)
        creds = json.loads(secret['SecretString'])
        return pymysql.connect(
            host=creds['host'],
            user=creds['username'],
            password=creds['password'],
            db=creds['dbname'],
            port=int(creds['port']),
            cursorclass=pymysql.cursors.DictCursor
        )

    def get_sqlalchemy_engine(self):
        secret = self.aws_clients.secrets_client.get_secret_value(SecretId=Config.SECRET_NAME)
        creds = json.loads(secret['SecretString'])
        return create_engine(f"mysql+pymysql://{creds['username']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['dbname']}")

class Hypnogram:
    def __init__(self, aws_clients: AWSClients, db_conn: DatabaseConnection):
        self.aws_clients = aws_clients
        self.db_conn = db_conn
        self.s3_ops = S3Operations(aws_clients)

    def list_subfolders(self, bucket: str) -> List[str]:
        """List all subfolders in the edf_files folder"""
        paginator = self.aws_clients.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=Config.EDF_FILES_PREFIX, Delimiter='/')

        subfolders = []
        for page in page_iterator:
            for prefix in page.get('CommonPrefixes', []):
                subfolders.append(prefix['Prefix'])
        return subfolders

    def list_files_in_subfolder(self, bucket: str, subfolder: str) -> List[str]:
        """List hypnogram files in a specific subfolder"""
        return self.s3_ops.list_files_without_extension(bucket, subfolder)

    def process_labeling_file(self, file_content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Process a labeling file to extract sleep stages and their time intervals."""
        labeling_df = pd.read_csv(io.StringIO(file_content), header=None, names=['data'])

        scan_date = None
        for idx, row in labeling_df.iterrows():
            if '[Rec. date:]' in row['data']:
                if idx + 1 < len(labeling_df):
                    scan_date = labeling_df.iloc[idx + 1]['data'].strip()
                    break
        
        stages = {}
        start_idx = None
        end_idx = None
        
        for idx, row in labeling_df.iterrows():
            if '[Events Channel_30 Hypnogr.]' in row['data']:
                start_idx = idx + 1
            elif start_idx is not None and row['data'].startswith('[Events Channel'):
                end_idx = idx
                break

        if end_idx is None:
            end_idx = len(labeling_df)

        for idx in range(start_idx, end_idx):
            line = labeling_df.iloc[idx]['data']
            if 'Start:' in line and 'Duration [ms]:' in line and 'Event:' in line and 'Light' not in line:
                start_time = line.split('Start:')[1].split(';')[0].strip()
                duration_ms = int(line.split('Duration [ms]:')[1].split(';')[0].strip())
                stage = line.split('Event:')[1].split(';')[0].strip()
                
                if stage not in stages:
                    stages[stage] = []
                
                hours = int(start_time.split(':')[0])
                date_str = scan_date if 18 <= hours < 24 else (datetime.strptime(scan_date, "%d.%m.%Y") + timedelta(days=1)).strftime("%d.%m.%Y")
                full_timestamp = datetime.strptime(f"{date_str} {start_time}", "%d.%m.%Y %H:%M:%S")
                
                stages[stage].append({
                    'start_time': full_timestamp,
                    'duration_ms': duration_ms
                })
        
        return stages

    def create_stage_intervals_df(self, stages: Dict[str, List[Dict[str, Any]]]) -> Optional[pd.DataFrame]:
        """Convert stages to a structured DataFrame."""
        stage_intervals = []
        for stage, intervals in stages.items():
            for interval in intervals:
                stage_intervals.append({
                    'Stage': stage,
                    'StartTime': interval['start_time'],
                    'DurationMs': interval['duration_ms']
                })
        
        if not stage_intervals:
            return None
            
        stage_intervals_df = pd.DataFrame(stage_intervals)
        stage_intervals_df['EndTime'] = stage_intervals_df.apply(
            lambda row: row['StartTime'] + timedelta(milliseconds=row['DurationMs']), axis=1)
        stage_intervals_df.drop(columns=['DurationMs'], inplace=True)
        return stage_intervals_df.sort_values('StartTime')

    def insert_sleep_stages(self, session_id: int, stage_intervals_df: Optional[pd.DataFrame]) -> None:
        """Insert sleep stages into database"""
        if stage_intervals_df is None or stage_intervals_df.empty:
            return
            
        stage_intervals_df['SessionID'] = session_id
        engine = self.db_conn.get_sqlalchemy_engine()
        stage_intervals_df.to_sql('SessionHypnogram', engine, if_exists='append', index=False)

class EDF:
    def __init__(self, aws_clients: AWSClients, db_conn: DatabaseConnection):
        self.aws_clients = aws_clients
        self.db_conn = db_conn
        self.s3_ops = S3Operations(aws_clients)

    def list_files_in_subfolder(self, bucket: str, subfolder: str) -> List[str]:
        """List EDF files in a specific subfolder"""
        return self.s3_ops.list_files_in_subfolder(bucket, subfolder, Config.EDF_SUFFIX)

    def load_and_process(self, bucket: str, key: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        local_path = f"/tmp/{key.split('/')[-1]}"
        print(f"Downloading file: {key} to {local_path}")
        self.aws_clients.s3_client.download_file(bucket, key, local_path)
        raw = mne.io.read_raw_edf(local_path, preload=False)
        raw.pick_channels(['Pulse', 'BR flow'])  # restrict first
        raw.load_data()  # now load only selected data
        os.remove(local_path)
        try:
            pulse_df = self.process_signal(raw, 'Pulse')
            br_df = self.process_signal(raw, 'BR flow')
            return pulse_df, br_df
        except Exception as e:
            print(f"Error reading channels: {e}")
            return None, None

    def process_signal(self, raw: mne.io.Raw, channel_name: str) -> Optional[pd.DataFrame]:
        data = raw.get_data(picks=channel_name)[0]
        times = raw.times
        start_time = raw.info['meas_date']
        df = pd.DataFrame({
            'time': [start_time + timedelta(seconds=t) for t in times],
            'original_signal': data
        })
        
        # Check if data is empty
        if df.empty or df['original_signal'].isna().all():
            return None
            
        mean = df['original_signal'].mean()
        std = df['original_signal'].std()
        df['filtered_signal'] = df['original_signal'].where(abs(df['original_signal'] - mean) <= 2 * std, np.nan)
        return df[['time', 'filtered_signal']]

    def compute_ref_value(self, df: Optional[pd.DataFrame], start_time: datetime) -> Optional[float]:
        if df is None:
            return None
            
        df['time'] = pd.to_datetime(df['time'], format='mixed').dt.tz_localize(None)
        start_time = pd.to_datetime(start_time).tz_localize(None)
        window = df[(df['time'] >= start_time) & (df['time'] < start_time + timedelta(seconds=60))]
        
        # Check if window is empty or all values are NaN
        if window.empty or window['filtered_signal'].isna().all():
            return None
            
        return window['filtered_signal'].median() / 60  # convert bpm to Hz

    def process_radar_signals(self, respiration_signals: List[Dict[str, Any]], pulse_df: Optional[pd.DataFrame], br_df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
        reference_data = []
        for signal in respiration_signals:
            signal_id = signal['RadarSignalID']
            start_time = signal['StartTime']
            hr = self.compute_ref_value(pulse_df, start_time)
            rr = self.compute_ref_value(br_df, start_time)
            reference_data.append({
                'RadarSignalID': signal_id,
                'HeartRate': hr,
                'RespirationRate': rr
            })
        return reference_data

    def insert_reference_results(self, reference_data: List[Dict[str, Any]]) -> None:
        if not reference_data:
            return
            
        df = pd.DataFrame(reference_data)
        df['ReferenceTypeID'] = Config.REFERENCE_TYPE_ID
        
        engine = self.db_conn.get_sqlalchemy_engine()
        df.to_sql('ReferenceResults', engine, if_exists='append', index=False)

class AnnonymizedVideo:
    def __init__(self, aws_clients: AWSClients, db_conn: DatabaseConnection):
        self.aws_clients = aws_clients
        self.db_conn = db_conn
        self.s3_ops = S3Operations(aws_clients)

    def list_files_in_subfolder(self, bucket: str, subfolder: str) -> List[str]:
        """List video files in a specific subfolder"""
        return self.s3_ops.list_files_in_subfolder(bucket, subfolder, Config.VIDEO_SUFFIX)

class DatabaseOperations:
    @staticmethod
    def is_file_processed(cursor: pymysql.cursors.DictCursor, key: str) -> bool:
        cursor.execute("SELECT 1 FROM SmlProcessedS3Files WHERE FilePath = %s LIMIT 1", (key,))
        return cursor.fetchone() is not None

    @staticmethod
    def mark_file_as_processed(cursor: pymysql.cursors.DictCursor, key: str) -> None:
        cursor.execute("INSERT INTO SmlProcessedS3Files (FilePath) VALUES (%s)", (key,))

    @staticmethod
    def get_session_id(cursor: pymysql.cursors.DictCursor, patient_name: str, session_date: str) -> Optional[int]:
        cursor.execute("""
            SELECT Session.ID
            FROM Session
            JOIN Patient ON Patient.ID = Session.PatientID
            WHERE Patient.PatientStudyName = %s AND DATE(Session.StartTime) = %s
            LIMIT 1
        """, (patient_name, session_date))
        row = cursor.fetchone()
        return row['ID'] if row else None

    @staticmethod
    def fetch_respiration_signals(cursor: pymysql.cursors.DictCursor, session_id: int) -> List[Dict[str, Any]]:
        cursor.execute("""
            SELECT ID as RadarSignalID, StartTime 
            FROM RadarSignal
            WHERE SessionID = %s
            AND RadarModeID = 3
        """, (session_id,))
        return cursor.fetchall()

    @staticmethod
    def update_session_edf_s3_path(cursor: pymysql.cursors.DictCursor, session_id: int, s3_path: str) -> None:
        cursor.execute("""
            UPDATE Session 
            SET S3FullPathReference = %s 
            WHERE ID = %s AND S3FullPathReference IS NULL
        """, (s3_path, session_id))

    @staticmethod
    def update_session_video_s3_path(cursor: pymysql.cursors.DictCursor, session_id: int, video_path: str) -> None:
        cursor.execute("""
            UPDATE Session 
            SET S3FullPathVideo = %s 
            WHERE ID = %s AND S3FullPathVideo IS NULL
        """, (video_path, session_id))

    @staticmethod
    def has_existing_reference_results(cursor: pymysql.cursors.DictCursor, session_id: int) -> bool:
        cursor.execute("""
            SELECT 1 FROM ReferenceResults rr
            JOIN RadarSignal rs ON rs.ID = rr.RadarSignalID
            WHERE rs.SessionID = %s
            LIMIT 1
        """, (session_id,))
        return cursor.fetchone() is not None

    @staticmethod
    def has_existing_hypnogram(cursor: pymysql.cursors.DictCursor, session_id: int) -> bool:
        cursor.execute("""
            SELECT 1 FROM SessionHypnogram
            WHERE SessionID = %s
            LIMIT 1
        """, (session_id,))
        return cursor.fetchone() is not None

class MainProcessor:
    def __init__(self):
        self.aws_clients = AWSClients()
        self.db_conn = DatabaseConnection(self.aws_clients)
        self.s3_ops = S3Operations(self.aws_clients)
        self.hypnogram = Hypnogram(self.aws_clients, self.db_conn)
        self.video = AnnonymizedVideo(self.aws_clients, self.db_conn)
        self.edf = EDF(self.aws_clients, self.db_conn)
        self.db_ops = DatabaseOperations()

    def validate_config(self) -> None:
        """Validate all required configuration"""
        required_configs = {
            'SECRET_NAME': Config.SECRET_NAME,
            'BUCKET_NAME': Config.BUCKET_NAME,
            'REGION': Config.REGION,
            'EDF_FILES_PREFIX': Config.EDF_FILES_PREFIX
        }
        
        missing_configs = [k for k, v in required_configs.items() if not v]
        if missing_configs:
            raise ValueError(f"Missing required configurations: {', '.join(missing_configs)}")

    @staticmethod
    def extract_session_info(folder_path: str, cursor: pymysql.cursors.DictCursor) -> Optional[SessionInfo]:
        """Extract session information from folder path"""
        try:
            # Extract the folder name (e.g., '20250122-0001' from 'edf_files/20250122-0001/')
            folder_name = folder_path.rstrip('/').split('/')[-1]
            session_part, patient_number = folder_name.split('-')
            patient_name = f"SL{patient_number}"
            
            # Convert session part to date format
            session_date = datetime.strptime(session_part, "%Y%m%d").strftime("%Y-%m-%d")
            
            # Get session ID
            session_id = DatabaseOperations.get_session_id(cursor, patient_name, session_date)
            
            return SessionInfo(
                patient_name=patient_name,
                session_date=session_date,
                session_id=session_id,
                folder_path=folder_path
            )
        except Exception as e:
            print(f"Error extracting session info from folder {folder_path}: {e}")
            return None

    def process_video(self, conn: pymysql.connections.Connection, bucket: str, key: str, session_info: SessionInfo) -> None:
        with conn.cursor() as cursor:
            if DatabaseOperations.is_file_processed(cursor, key):
                print(f"Skipped already processed video: {key}")
                return

            try:
                if not session_info.session_id:
                    print(f"No session found for video file {key}")
                    return

                s3_path = f"{bucket}/{key}"
                DatabaseOperations.update_session_video_s3_path(cursor, session_info.session_id, s3_path)
                DatabaseOperations.mark_file_as_processed(cursor, key)
                conn.commit()
                print(f"Updated video path for session {session_info.session_id}")

            except Exception as e:
                print(f"Error processing video file {key}: {e}")

    def process_edf_file(self, conn: pymysql.connections.Connection, bucket: str, key: str, session_info: SessionInfo) -> None:
        with conn.cursor() as cursor:
            if not session_info.session_id:
                print(f"No session found for EDF file {key}")
                return

            if DatabaseOperations.is_file_processed(cursor, key):
                print(f"Skipped already processed EDF: {key}")
                return

            # Check if reference results already exist
            if self.db_ops.has_existing_reference_results(cursor, session_info.session_id):
                print(f"Reference results already exist for session {session_info.session_id}, skipping EDF processing")
                self.db_ops.mark_file_as_processed(cursor, key)
                conn.commit()
                return

            # Update S3 path in Session table if it's NULL
            s3_path = f"{bucket}/{key}"
            self.db_ops.update_session_edf_s3_path(cursor, session_info.session_id, s3_path)
                
            respiration_signals = self.db_ops.fetch_respiration_signals(cursor, session_info.session_id)
            if not respiration_signals:
                print(f"No respiration signals found for session {session_info.session_id}")
                return

            pulse_df, br_df = self.edf.load_and_process(bucket, key)
            if pulse_df is None or br_df is None:
                return

            reference_data = self.edf.process_radar_signals(respiration_signals, pulse_df, br_df)
            self.edf.insert_reference_results(reference_data)
            print(f"Saved {len(reference_data)} reference results for patient {session_info.patient_name} on {session_info.session_date}, session {session_info.session_id}")

            self.db_ops.mark_file_as_processed(cursor, key)
            conn.commit()

    def process_hypnogram_file(self, conn: pymysql.connections.Connection, bucket: str, key: str, session_info: SessionInfo) -> None:
        with conn.cursor() as cursor:
            if self.db_ops.is_file_processed(cursor, key):
                print(f"Skipped already processed hypnogram: {key}")
                return

            try:
                if not session_info.session_id:
                    print(f"No session found for labeling file {key}")
                    return

                # Check if hypnogram data already exists
                if self.db_ops.has_existing_hypnogram(cursor, session_info.session_id):
                    print(f"Hypnogram data already exists for session {session_info.session_id}, skipping hypnogram processing")
                    self.db_ops.mark_file_as_processed(cursor, key)
                    conn.commit()
                    return

                response = self.aws_clients.s3_client.get_object(Bucket=bucket, Key=key)
                file_content = response['Body'].read().decode('utf-8')
                
                stages = self.hypnogram.process_labeling_file(file_content)
                stage_intervals_df = self.hypnogram.create_stage_intervals_df(stages)
                self.hypnogram.insert_sleep_stages(session_info.session_id, stage_intervals_df)
                print(f"Saved sleep stages for session {session_info.session_id}")

                self.db_ops.mark_file_as_processed(cursor, key)
                conn.commit()

            except Exception as e:
                print(f"Error processing hypnogram file {key}: {e}")

    def process_subfolder(self, conn: pymysql.connections.Connection, bucket: str, subfolder: str) -> None:
        """Process all files in a subfolder"""
        print(f"Processing subfolder: {subfolder}")
        
        with conn.cursor() as cursor:
            # Extract session info once for the folder
            session_info = self.extract_session_info(subfolder, cursor)
            if not session_info:
                print(f"Could not extract session info from folder {subfolder}, skipping")
                return
                
            # Get all unprocessed files in this subfolder
            unprocessed_files = self.s3_ops.get_unprocessed_files(bucket, subfolder, cursor)
            
            # Process video files
            for key in unprocessed_files['video']:
                self.process_video(conn, bucket, key, session_info)
            
            # Process hypnogram files first
            for key in unprocessed_files['hypno']:
                self.process_hypnogram_file(conn, bucket, key, session_info)
                
            # Then process EDF files
            for key in unprocessed_files['edf']:
                self.process_edf_file(conn, bucket, key, session_info)

    def process_all_folders(self) -> Dict[str, Any]:
        """Main processing function that handles all folders"""
        try:
            self.validate_config()
            
            conn = self.db_conn.get_db_connection()
            processed_folders = 0
            error_folders = []

            try:
                # Get all subfolders
                subfolders = self.s3_ops.list_subfolders(Config.BUCKET_NAME, Config.EDF_FILES_PREFIX)
                
                # Process each subfolder sequentially
                for subfolder in subfolders:
                    try:
                        self.process_subfolder(conn, Config.BUCKET_NAME, subfolder)
                        processed_folders += 1
                    except Exception as e:
                        print(f"Error processing subfolder {subfolder}: {str(e)}")
                        error_folders.append({"folder": subfolder, "error": str(e)})

                return {
                    "statusCode": 200,
                    "body": {
                        "message": "Processing completed",
                        "processed_folders": processed_folders,
                        "total_folders": len(subfolders),
                        "error_folders": error_folders
                    }
                }

            finally:
                conn.close()

        except Exception as e:
            print(f"Fatal error in processing: {str(e)}")
            return {
                "statusCode": 500,
                "body": {
                    "message": "Fatal error in processing",
                    "error": str(e)
                }
            }

# Initialize processor once when the Lambda container starts
processor = MainProcessor()

def handler(event: Optional[Dict[str, Any]], context: Optional[Any]) -> Dict[str, Any]:
    """AWS Lambda handler function
    
    Args:
        event: AWS Lambda event object
        context: AWS Lambda context object
        
    Returns:
        Dict containing statusCode and processing results
    """
    try:
        # Log the incoming event if needed
        if event:
            print(f"Processing event: {json.dumps(event)}")
            
        # Process all folders
        return processor.process_all_folders()
        
    except Exception as e:
        print(f"Error in Lambda handler: {str(e)}")
        return {
            "statusCode": 500,
            "body": {
                "message": "Error in Lambda handler",
                "error": str(e)
            }
        }

if __name__ == "__main__":
    # For local testing
    result = handler(None, None)
    print(json.dumps(result, indent=2))