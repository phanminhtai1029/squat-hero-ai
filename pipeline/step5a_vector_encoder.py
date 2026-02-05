"""
Step 5a: Vector Encoder
=======================
Converts pose keypoints to embedding vectors using GNN.
"""

import numpy as np
import torch
from typing import Optional, Union
from pathlib import Path

# Import pose result from step3
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.step3_pose_estimation import PoseResult


class VectorEncoder:
    """
    Encode pose keypoints into embedding vectors.
    
    Uses a lightweight GNN to convert 17 COCO keypoints 
    into a 128D L2-normalized embedding vector.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_dim: int = 128,
        device: str = None
    ):
        """
        Initialize vector encoder.
        
        Args:
            model_path: Path to trained model checkpoint (optional)
            embed_dim: Dimension of embedding vectors
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load model
        from ml.models.pose_gnn_encoder import SimplePoseGNN
        self.model = SimplePoseGNN(embed_dim=embed_dim)
        
        if model_path and Path(model_path).exists():
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained encoder from: {model_path}")
        else:
            # Use randomly initialized weights (for testing)
            print("⚠️  Using untrained encoder (random weights)")
            print("   For best results, train the model first!")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"Vector Encoder initialized:")
        print(f"  Device: {device}")
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Parameters: {self.model.get_num_params():,}")
    
    def encode(
        self, 
        pose_result: Union[PoseResult, np.ndarray]
    ) -> np.ndarray:
        """
        Encode pose to embedding vector.
        
        Args:
            pose_result: PoseResult or numpy array (17, 2) or (17, 4)
        
        Returns:
            embedding: (128,) L2-normalized vector
        """
        # Extract keypoints
        if isinstance(pose_result, PoseResult):
            keypoints = pose_result.to_numpy()[:, :2]  # (17, 2) x, y only
        else:
            keypoints = pose_result[:, :2] if pose_result.shape[1] > 2 else pose_result
        
        # Convert to tensor
        kpts_tensor = torch.from_numpy(keypoints).float().to(self.device)
        
        # Encode
        with torch.no_grad():
            embedding = self.model(kpts_tensor)  # (128,)
        
        return embedding.cpu().numpy()
    
    def encode_batch(
        self, 
        pose_results: list
    ) -> np.ndarray:
        """
        Encode multiple poses in batch.
        
        Args:
            pose_results: List of PoseResult or numpy arrays
        
        Returns:
            embeddings: (N, 128) array
        """
        # Extract all keypoints
        keypoints_list = []
        for pr in pose_results:
            if isinstance(pr, PoseResult):
                kpts = pr.to_numpy()[:, :2]
            else:
                kpts = pr[:, :2] if pr.shape[1] > 2 else pr
            keypoints_list.append(kpts)
        
        # Stack to batch
        batch_kpts = np.stack(keypoints_list)  # (N, 17, 2)
        batch_tensor = torch.from_numpy(batch_kpts).float().to(self.device)
        
        # Encode batch
        with torch.no_grad():
            embeddings = self.model(batch_tensor)  # (N, 128)
        
        return embeddings.cpu().numpy()


if __name__ == '__main__':
    # Test encoder
    print("Testing VectorEncoder...")
    
    encoder = VectorEncoder(embed_dim=128)
    
    # Test single pose
    dummy_keypoints = np.random.randn(17, 2).astype(np.float32)
    embedding = encoder.encode(dummy_keypoints)
    
    print(f"\nSingle pose encoding:")
    print(f"  Input shape: {dummy_keypoints.shape}")
    print(f"  Output shape: {embedding.shape}")
    print(f"  L2 norm: {np.linalg.norm(embedding):.4f}")
    
    # Test batch
    dummy_batch = [np.random.randn(17, 2).astype(np.float32) for _ in range(5)]
    batch_embeddings = encoder.encode_batch(dummy_batch)
    
    print(f"\nBatch encoding:")
    print(f"  Batch size: {len(dummy_batch)}")
    print(f"  Output shape: {batch_embeddings.shape}")
    print(f"  L2 norms: {np.linalg.norm(batch_embeddings, axis=1)}")
    
    # Benchmark speed
    import time
    n_iters = 100
    
    start = time.time()
    for _ in range(n_iters):
        _ = encoder.encode(dummy_keypoints)
    elapsed = (time.time() - start) / n_iters * 1000
    
    print(f"\nInference speed:")
    print(f"  {elapsed:.2f}ms per sample")
    print(f"  {1000/elapsed:.1f} FPS")
    
    print("\n✅ Encoder test passed!")
