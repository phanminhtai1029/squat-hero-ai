"""
Step 5b: Vector Matcher
=======================
Matches pose embeddings using vector similarity search.

Supports multiple matching algorithms:
1. FAISS - Fast approximate nearest neighbor (GPU/CPU)
2. Cosine Similarity - Exact but slower
3. Euclidean Distance - Simple L2 distance
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path


@dataclass
class MatchResult:
    """Pose matching result."""
    pose_name: str
    display_name: str
    similarity: float
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    top_k_matches: Optional[List[Tuple[str, float]]] = None


class VectorMatcher:
    """
    Match pose embeddings using vector similarity.
    
    Supports:
    - FAISS index for fast approximate search
    - Fallback to exact cosine similarity
    """
    
    def __init__(
        self,
        database_path: str = "data/pose_vectors.npz",
        use_faiss: bool = True,
        k: int = 5
    ):
        """
        Initialize vector matcher.
        
        Args:
            database_path: Path to vector database (.npz file)
            use_faiss: Use FAISS for fast search (if available)
            k: Number of nearest neighbors to retrieve
        """
        self.k = k
        self.use_faiss = use_faiss
        
        # Load database
        self._load_database(database_path)
        
        # Initialize search index
        if use_faiss:
            try:
                import faiss
                self.faiss = faiss
                self._build_faiss_index()
                print(f"Using FAISS index (fast approximate search)")
            except ImportError:
                print("⚠️  FAISS not available, using exact cosine similarity")
                self.use_faiss = False
                self.faiss = None
        else:
            self.faiss = None
    
    def _load_database(self, database_path: str):
        """Load vector database from .npz file."""
        db_path = Path(database_path)
        
        if not db_path.exists():
            raise FileNotFoundError(
                f"Vector database not found: {database_path}\n"
                f"Please run: python ml/build_vector_database.py"
            )
        
        # Load numpy archive
        data = np.load(database_path, allow_pickle=True)
        
        self.embeddings = data['embeddings']  # (N, embed_dim)
        self.labels = data['labels']  # (N,) pose names
        self.metadata = data['metadata'].item()  # dict with pose info
        
        # Build label to display name mapping
        self.label_to_display = {}
        for pose_name, info in self.metadata.items():
            self.label_to_display[pose_name] = info.get('display_name', pose_name)
        
        print(f"Loaded vector database:")
        print(f"  Embeddings: {self.embeddings.shape}")
        print(f"  Num poses: {len(set(self.labels))}")
        print(f"  Total samples: {len(self.labels)}")
        print(f"  Embedding dim: {self.embeddings.shape[1]}")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        embed_dim = self.embeddings.shape[1]  # Shape[1] is embedding dimension (128)
        
        # L2 distance (Euclidean)
        # For normalized vectors, L2 distance ≈ 2 - 2*cosine_similarity
        self.index = self.faiss.IndexFlatL2(embed_dim)
        
        # Add all embeddings
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"  FAISS index built with {self.index.ntotal} vectors")
    
    def _cosine_similarity(
        self, 
        query: np.ndarray, 
        database: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and database.
        
        Args:
            query: (embed_dim,) query vector (L2 normalized)
            database: (N, embed_dim) database vectors (L2 normalized)
        
        Returns:
            similarities: (N,) cosine similarities
        """
        # For L2-normalized vectors: cosine_sim = dot product
        return database @ query
    
    def match(
        self, 
        embedding: np.ndarray,
        return_top_k: bool = False
    ) -> MatchResult:
        """
        Find best matching pose for embedding.
        
        Args:
            embedding: (embed_dim,) L2-normalized embedding vector
            return_top_k: Return top-k matches in result
        
        Returns:
            MatchResult with pose name and confidence
        """
        embedding = embedding.astype('float32')
        
        # Search for k nearest neighbors
        if self.use_faiss and self.faiss is not None:
            # FAISS search (returns L2 distances)
            distances, indices = self.index.search(
                embedding.reshape(1, -1), 
                self.k
            )
            distances = distances[0]  # (k,)
            indices = indices[0]  # (k,)
            
            # Convert L2 distance to similarity
            # For normalized vectors: sim = 1 - dist²/2
            similarities = 1 - (distances ** 2) / 2
            
        else:
            # Exact cosine similarity
            similarities = self._cosine_similarity(embedding, self.embeddings)
            
            # Get top-k
            indices = np.argsort(similarities)[::-1][:self.k]
            similarities = similarities[indices]
        
        # Get corresponding pose names
        top_poses = [self.labels[i] for i in indices]
        top_sims = similarities.tolist()
        
        # Majority vote among top-k
        pose_counts = Counter(top_poses)
        best_pose = pose_counts.most_common(1)[0][0]
        
        # Average similarity for best pose
        best_pose_sims = [s for p, s in zip(top_poses, top_sims) if p == best_pose]
        avg_similarity = float(np.mean(best_pose_sims))
        
        # Confidence scoring
        confidence = self._compute_confidence(avg_similarity)
        
        # Prepare result
        result = MatchResult(
            pose_name=best_pose,
            display_name=self.label_to_display.get(best_pose, best_pose),
            similarity=avg_similarity,
            confidence=confidence
        )
        
        if return_top_k:
            result.top_k_matches = [
                (self.label_to_display.get(p, p), s) 
                for p, s in zip(top_poses, top_sims)
            ]
        
        return result
    
    def _compute_confidence(self, similarity: float) -> str:
        """Convert similarity score to confidence level."""
        if similarity >= 0.85:
            return 'HIGH'
        elif similarity >= 0.70:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        pose_counts = Counter(self.labels)
        
        return {
            'total_samples': len(self.labels),
            'num_poses': len(pose_counts),
            'pose_distribution': dict(pose_counts),
            'embedding_dim': self.embeddings.shape[1],
            'index_type': 'FAISS' if self.use_faiss else 'Cosine Similarity'
        }


if __name__ == '__main__':
    # Test matcher (will fail without database)
    print("Testing VectorMatcher...")
    print("⚠️  This will fail without a vector database")
    print("   Run: python ml/build_vector_database.py first\n")
    
    try:
        matcher = VectorMatcher(database_path="data/pose_vectors.npz")
        
        # Test with dummy embedding
        dummy_embedding = np.random.randn(128).astype(np.float32)
        dummy_embedding = dummy_embedding / np.linalg.norm(dummy_embedding)
        
        result = matcher.match(dummy_embedding, return_top_k=True)
        
        print(f"\nMatch result:")
        print(f"  Pose: {result.display_name}")
        print(f"  Similarity: {result.similarity:.3f}")
        print(f"  Confidence: {result.confidence}")
        print(f"\nTop-{matcher.k} matches:")
        for i, (pose, sim) in enumerate(result.top_k_matches, 1):
            print(f"    {i}. {pose}: {sim:.3f}")
        
        # Show statistics
        stats = matcher.get_statistics()
        print(f"\nDatabase statistics:")
        for key, value in stats.items():
            if key != 'pose_distribution':
                print(f"  {key}: {value}")
        
        print("\n✅ Matcher test passed!")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nTo create vector database:")
        print("  1. Run: python ml/build_vector_database.py")
        print("  2. Then run this test again")
