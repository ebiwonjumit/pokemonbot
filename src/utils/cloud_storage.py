"""
Cloud storage utilities for Pokemon RL Bot.
Handles model uploads, downloads, and synchronization with cloud providers.
"""

import os
import boto3
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import hashlib
import tempfile
import logging
from dataclasses import dataclass

from .logger import get_logger

logger = get_logger(__name__)

try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("Google Cloud Storage not available - install google-cloud-storage")


@dataclass
class ModelMetadata:
    """Metadata for uploaded models."""
    model_id: str
    version: str
    timestamp: datetime
    file_size: int
    checksum: str
    training_episodes: int
    performance_metrics: Dict[str, float]
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class CloudStorageBase:
    """Base class for cloud storage providers."""
    
    def __init__(self, bucket_name: str):
        """Initialize cloud storage."""
        self.bucket_name = bucket_name
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    def upload_model(
        self,
        local_path: str,
        remote_path: str,
        metadata: Optional[ModelMetadata] = None
    ) -> bool:
        """Upload model to cloud storage."""
        raise NotImplementedError
    
    def download_model(self, remote_path: str, local_path: str) -> bool:
        """Download model from cloud storage."""
        raise NotImplementedError
    
    def list_models(self, prefix: str = "models/") -> List[str]:
        """List available models in cloud storage."""
        raise NotImplementedError
    
    def delete_model(self, remote_path: str) -> bool:
        """Delete model from cloud storage."""
        raise NotImplementedError
    
    def model_exists(self, remote_path: str) -> bool:
        """Check if model exists in cloud storage."""
        raise NotImplementedError


class AWSStorage(CloudStorageBase):
    """AWS S3 storage provider."""
    
    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-west-2"
    ):
        """
        Initialize AWS S3 storage.
        
        Args:
            bucket_name: S3 bucket name
            aws_access_key_id: AWS access key (optional, can use env vars)
            aws_secret_access_key: AWS secret key (optional, can use env vars)
            region_name: AWS region
        """
        super().__init__(bucket_name)
        
        # Initialize S3 client
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        self.s3_client = session.client('s3')
        self.region_name = region_name
        
        # Verify bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.info(f"Connected to S3 bucket: {bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to S3 bucket {bucket_name}: {e}")
            raise
    
    def upload_model(
        self,
        local_path: str,
        remote_path: str,
        metadata: Optional[ModelMetadata] = None
    ) -> bool:
        """Upload model to S3."""
        try:
            # Prepare metadata for S3
            s3_metadata = {}
            if metadata:
                s3_metadata = {
                    'model-id': metadata.model_id,
                    'version': metadata.version,
                    'timestamp': metadata.timestamp.isoformat(),
                    'training-episodes': str(metadata.training_episodes),
                    'checksum': metadata.checksum
                }
                
                if metadata.description:
                    s3_metadata['description'] = metadata.description
                
                if metadata.tags:
                    s3_metadata['tags'] = ','.join(metadata.tags)
            
            # Upload file
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                remote_path,
                ExtraArgs={'Metadata': s3_metadata}
            )
            
            # Upload metadata as separate JSON file
            if metadata:
                metadata_path = remote_path + ".metadata.json"
                metadata_dict = {
                    'model_id': metadata.model_id,
                    'version': metadata.version,
                    'timestamp': metadata.timestamp.isoformat(),
                    'file_size': metadata.file_size,
                    'checksum': metadata.checksum,
                    'training_episodes': metadata.training_episodes,
                    'performance_metrics': metadata.performance_metrics,
                    'description': metadata.description,
                    'tags': metadata.tags
                }
                
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                    json.dump(metadata_dict, f, indent=2)
                    temp_path = f.name
                
                try:
                    self.s3_client.upload_file(temp_path, self.bucket_name, metadata_path)
                finally:
                    os.unlink(temp_path)
            
            self.logger.info(f"Uploaded model to S3: {remote_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload model to S3: {e}")
            return False
    
    def download_model(self, remote_path: str, local_path: str) -> bool:
        """Download model from S3."""
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(self.bucket_name, remote_path, local_path)
            
            self.logger.info(f"Downloaded model from S3: {remote_path} -> {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download model from S3: {e}")
            return False
    
    def list_models(self, prefix: str = "models/") -> List[str]:
        """List available models in S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            models = []
            for obj in response['Contents']:
                key = obj['Key']
                # Skip metadata files
                if not key.endswith('.metadata.json'):
                    models.append(key)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models from S3: {e}")
            return []
    
    def delete_model(self, remote_path: str) -> bool:
        """Delete model from S3."""
        try:
            # Delete model file
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=remote_path)
            
            # Delete metadata file if exists
            metadata_path = remote_path + ".metadata.json"
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=metadata_path)
            except:
                pass  # Metadata file might not exist
            
            self.logger.info(f"Deleted model from S3: {remote_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model from S3: {e}")
            return False
    
    def model_exists(self, remote_path: str) -> bool:
        """Check if model exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_path)
            return True
        except:
            return False
    
    def get_model_metadata(self, remote_path: str) -> Optional[ModelMetadata]:
        """Get model metadata from S3."""
        try:
            metadata_path = remote_path + ".metadata.json"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
                temp_path = f.name
            
            try:
                self.s3_client.download_file(self.bucket_name, metadata_path, temp_path)
                
                with open(temp_path, 'r') as f:
                    metadata_dict = json.load(f)
                
                return ModelMetadata(
                    model_id=metadata_dict['model_id'],
                    version=metadata_dict['version'],
                    timestamp=datetime.fromisoformat(metadata_dict['timestamp']),
                    file_size=metadata_dict['file_size'],
                    checksum=metadata_dict['checksum'],
                    training_episodes=metadata_dict['training_episodes'],
                    performance_metrics=metadata_dict['performance_metrics'],
                    description=metadata_dict.get('description'),
                    tags=metadata_dict.get('tags')
                )
                
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            self.logger.error(f"Failed to get model metadata from S3: {e}")
            return None


class GCPStorage(CloudStorageBase):
    """Google Cloud Storage provider."""
    
    def __init__(
        self,
        bucket_name: str,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize Google Cloud Storage.
        
        Args:
            bucket_name: GCS bucket name
            credentials_path: Path to service account JSON file
            project_id: GCP project ID
        """
        super().__init__(bucket_name)
        
        if not GCS_AVAILABLE:
            raise ImportError("Google Cloud Storage not available")
        
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Initialize GCS client
        self.client = gcs.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
        
        # Verify bucket exists
        try:
            if not self.bucket.exists():
                raise ValueError(f"Bucket {bucket_name} does not exist")
            self.logger.info(f"Connected to GCS bucket: {bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to GCS bucket {bucket_name}: {e}")
            raise
    
    def upload_model(
        self,
        local_path: str,
        remote_path: str,
        metadata: Optional[ModelMetadata] = None
    ) -> bool:
        """Upload model to GCS."""
        try:
            blob = self.bucket.blob(remote_path)
            
            # Set metadata
            if metadata:
                blob.metadata = {
                    'model_id': metadata.model_id,
                    'version': metadata.version,
                    'timestamp': metadata.timestamp.isoformat(),
                    'training_episodes': str(metadata.training_episodes),
                    'checksum': metadata.checksum
                }
            
            # Upload file
            blob.upload_from_filename(local_path)
            
            # Upload metadata as separate file
            if metadata:
                metadata_blob = self.bucket.blob(remote_path + ".metadata.json")
                metadata_dict = {
                    'model_id': metadata.model_id,
                    'version': metadata.version,
                    'timestamp': metadata.timestamp.isoformat(),
                    'file_size': metadata.file_size,
                    'checksum': metadata.checksum,
                    'training_episodes': metadata.training_episodes,
                    'performance_metrics': metadata.performance_metrics,
                    'description': metadata.description,
                    'tags': metadata.tags
                }
                
                metadata_blob.upload_from_string(
                    json.dumps(metadata_dict, indent=2),
                    content_type='application/json'
                )
            
            self.logger.info(f"Uploaded model to GCS: {remote_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload model to GCS: {e}")
            return False
    
    def download_model(self, remote_path: str, local_path: str) -> bool:
        """Download model from GCS."""
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            blob = self.bucket.blob(remote_path)
            blob.download_to_filename(local_path)
            
            self.logger.info(f"Downloaded model from GCS: {remote_path} -> {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download model from GCS: {e}")
            return False
    
    def list_models(self, prefix: str = "models/") -> List[str]:
        """List available models in GCS."""
        try:
            blobs = self.client.list_blobs(self.bucket, prefix=prefix)
            
            models = []
            for blob in blobs:
                # Skip metadata files
                if not blob.name.endswith('.metadata.json'):
                    models.append(blob.name)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models from GCS: {e}")
            return []
    
    def delete_model(self, remote_path: str) -> bool:
        """Delete model from GCS."""
        try:
            # Delete model file
            blob = self.bucket.blob(remote_path)
            blob.delete()
            
            # Delete metadata file if exists
            try:
                metadata_blob = self.bucket.blob(remote_path + ".metadata.json")
                metadata_blob.delete()
            except:
                pass
            
            self.logger.info(f"Deleted model from GCS: {remote_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model from GCS: {e}")
            return False
    
    def model_exists(self, remote_path: str) -> bool:
        """Check if model exists in GCS."""
        try:
            blob = self.bucket.blob(remote_path)
            return blob.exists()
        except:
            return False


class ModelManager:
    """High-level model management with cloud storage."""
    
    def __init__(
        self,
        storage_provider: CloudStorageBase,
        local_models_dir: str = "models"
    ):
        """
        Initialize model manager.
        
        Args:
            storage_provider: Cloud storage provider instance
            local_models_dir: Local directory for models
        """
        self.storage = storage_provider
        self.local_dir = Path(local_models_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(f"{__name__}.ModelManager")
    
    def save_and_upload_model(
        self,
        model_path: str,
        model_id: str,
        version: str,
        training_episodes: int,
        performance_metrics: Dict[str, float],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Save model locally and upload to cloud storage.
        
        Args:
            model_path: Local path to model file
            model_id: Unique model identifier
            version: Model version
            training_episodes: Number of training episodes
            performance_metrics: Performance metrics dict
            description: Optional description
            tags: Optional tags list
            
        Returns:
            bool: True if successful
        """
        try:
            # Calculate file checksum
            checksum = self._calculate_checksum(model_path)
            file_size = os.path.getsize(model_path)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                version=version,
                timestamp=datetime.now(),
                file_size=file_size,
                checksum=checksum,
                training_episodes=training_episodes,
                performance_metrics=performance_metrics,
                description=description,
                tags=tags
            )
            
            # Generate remote path
            remote_path = f"models/{model_id}/{version}/{Path(model_path).name}"
            
            # Upload to cloud storage
            success = self.storage.upload_model(model_path, remote_path, metadata)
            
            if success:
                self.logger.info(f"Model uploaded successfully: {model_id} v{version}")
                
                # Save metadata locally
                self._save_local_metadata(model_path, metadata)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to save and upload model: {e}")
            return False
    
    def download_model(
        self,
        model_id: str,
        version: str,
        filename: str = "model.zip"
    ) -> Optional[str]:
        """
        Download model from cloud storage.
        
        Args:
            model_id: Model identifier
            version: Model version
            filename: Model filename
            
        Returns:
            Optional[str]: Local path to downloaded model or None if failed
        """
        try:
            remote_path = f"models/{model_id}/{version}/{filename}"
            local_path = self.local_dir / model_id / version / filename
            
            # Ensure local directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            success = self.storage.download_model(remote_path, str(local_path))
            
            if success:
                self.logger.info(f"Model downloaded: {model_id} v{version}")
                return str(local_path)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to download model: {e}")
            return None
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models in cloud storage."""
        try:
            model_files = self.storage.list_models()
            
            models = []
            processed_models = set()
            
            for file_path in model_files:
                # Parse model path: models/{model_id}/{version}/{filename}
                parts = file_path.split('/')
                if len(parts) >= 4 and parts[0] == 'models':
                    model_id = parts[1]
                    version = parts[2]
                    filename = parts[3]
                    
                    model_key = f"{model_id}:{version}"
                    if model_key not in processed_models:
                        models.append({
                            'model_id': model_id,
                            'version': version,
                            'filename': filename,
                            'remote_path': file_path
                        })
                        processed_models.add(model_key)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def delete_model(self, model_id: str, version: str, filename: str = "model.zip") -> bool:
        """Delete model from cloud storage."""
        try:
            remote_path = f"models/{model_id}/{version}/{filename}"
            success = self.storage.delete_model(remote_path)
            
            if success:
                self.logger.info(f"Model deleted: {model_id} v{version}")
                
                # Also delete local copy if exists
                local_path = self.local_dir / model_id / version / filename
                if local_path.exists():
                    local_path.unlink()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete model: {e}")
            return False
    
    def sync_local_models(self) -> int:
        """Sync local models to cloud storage."""
        synced_count = 0
        
        try:
            # Find all local model files
            for model_file in self.local_dir.rglob("*.zip"):
                # Parse local path structure
                relative_path = model_file.relative_to(self.local_dir)
                parts = relative_path.parts
                
                if len(parts) >= 3:  # model_id/version/filename
                    model_id = parts[0]
                    version = parts[1]
                    filename = parts[2]
                    
                    remote_path = f"models/{model_id}/{version}/{filename}"
                    
                    # Check if already exists in cloud
                    if not self.storage.model_exists(remote_path):
                        # Load local metadata if available
                        metadata = self._load_local_metadata(str(model_file))
                        
                        # Upload to cloud
                        if self.storage.upload_model(str(model_file), remote_path, metadata):
                            synced_count += 1
                            self.logger.info(f"Synced model: {model_id} v{version}")
            
            self.logger.info(f"Synced {synced_count} models to cloud storage")
            return synced_count
            
        except Exception as e:
            self.logger.error(f"Failed to sync local models: {e}")
            return synced_count
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _save_local_metadata(self, model_path: str, metadata: ModelMetadata):
        """Save metadata alongside local model file."""
        metadata_path = model_path + ".metadata.json"
        
        metadata_dict = {
            'model_id': metadata.model_id,
            'version': metadata.version,
            'timestamp': metadata.timestamp.isoformat(),
            'file_size': metadata.file_size,
            'checksum': metadata.checksum,
            'training_episodes': metadata.training_episodes,
            'performance_metrics': metadata.performance_metrics,
            'description': metadata.description,
            'tags': metadata.tags
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _load_local_metadata(self, model_path: str) -> Optional[ModelMetadata]:
        """Load metadata from local file."""
        metadata_path = model_path + ".metadata.json"
        
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            return ModelMetadata(
                model_id=metadata_dict['model_id'],
                version=metadata_dict['version'],
                timestamp=datetime.fromisoformat(metadata_dict['timestamp']),
                file_size=metadata_dict['file_size'],
                checksum=metadata_dict['checksum'],
                training_episodes=metadata_dict['training_episodes'],
                performance_metrics=metadata_dict['performance_metrics'],
                description=metadata_dict.get('description'),
                tags=metadata_dict.get('tags')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load local metadata: {e}")
            return None


def create_storage_provider(provider: str, **kwargs) -> CloudStorageBase:
    """
    Factory function to create storage provider.
    
    Args:
        provider: Storage provider name ('aws' or 'gcp')
        **kwargs: Provider-specific arguments
        
    Returns:
        CloudStorageBase: Storage provider instance
    """
    if provider.lower() == 'aws':
        return AWSStorage(**kwargs)
    elif provider.lower() == 'gcp':
        return GCPStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage provider: {provider}")


if __name__ == "__main__":
    # Test cloud storage functionality
    print("Testing cloud storage...")
    
    # Test with dummy data (requires actual credentials)
    try:
        # Example with AWS (requires credentials)
        # storage = AWSStorage("pokemon-rl-models")
        # manager = ModelManager(storage)
        
        # Create dummy model file for testing
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            f.write(b"dummy model data")
            test_model_path = f.name
        
        print(f"Created test model: {test_model_path}")
        
        # Example usage:
        # success = manager.save_and_upload_model(
        #     test_model_path,
        #     model_id="pokemon_rl_v1",
        #     version="1.0.0",
        #     training_episodes=1000,
        #     performance_metrics={"avg_reward": 100.5, "badges": 8},
        #     description="Test model upload"
        # )
        
        # models = manager.list_available_models()
        # print(f"Available models: {models}")
        
        # Clean up
        os.unlink(test_model_path)
        print("Cloud storage test completed!")
        
    except Exception as e:
        print(f"Cloud storage test failed: {e}")
        print("Note: Actual cloud credentials required for full testing")
