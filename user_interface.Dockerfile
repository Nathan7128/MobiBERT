# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY mobiBERT/user_interface/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY mobiBERT/ mobiBERT/

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "mobiBERT/user_interface/UI.py"]