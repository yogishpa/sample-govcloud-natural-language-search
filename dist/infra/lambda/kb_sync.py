"""Lambda function to trigger Bedrock Knowledge Base ingestion sync."""

import os
import boto3

KB_ID = os.environ["KNOWLEDGE_BASE_ID"]
DS_ID = os.environ["DATA_SOURCE_ID"]

client = boto3.client("bedrock-agent")


def handler(event, context):
    response = client.start_ingestion_job(
        knowledgeBaseId=KB_ID,
        dataSourceId=DS_ID,
    )
    job = response["ingestionJob"]
    print(f"Started ingestion job {job['ingestionJobId']} - status: {job['status']}")
    return {"ingestionJobId": job["ingestionJobId"], "status": job["status"]}
