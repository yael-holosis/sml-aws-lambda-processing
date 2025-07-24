FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy all files in ./src
COPY src/* ${LAMBDA_TASK_ROOT}/

# Set proper permissions
RUN chmod 644 ${LAMBDA_TASK_ROOT}/*.py

# Set the CMD to your handler.
CMD [ "main.handler" ]