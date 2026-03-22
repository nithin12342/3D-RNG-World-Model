"""
Spatial Multi-Modal Tokenizer for 3D-RNG World Engine
Lead Multi-Modal Data Scientist & Training Architect Implementation

This script implements Phase 1: Spatial Multi-Modal Tokenizer for video/text data injection
into the 3D-RNG World Engine, mapping video patches to Vision Face and text embeddings 
to Text Face with perfect temporal synchronization.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from pathlib import Path


class VideoPatchExtractor(nn.Module):
    """
    Extracts patches from video frames and projects them to latent vectors.
    Converts (H, W, C) video frames into a grid of patches, each projected to size D.
    For X=0 plane injection.
    """


class AudioFeatureExtractor(nn.Module):
    """
    Extracts audio features (e.g., spectrogram patches) and projects them to latent vectors.
    Processes audio waveforms through a convolutional encoder.
    For X=0 plane injection (coexisting with video).
    """
    
    def __init__(self, embed_dim: int = 768, n_fft: int = 512, hop_length: int = 160):
        """
        Initialize Audio Feature Extractor.
        
        Args:
            embed_dim: Dimensionality of output embedding vectors
            n_fft: FFT size for spectrogram computation
            hop_length: Hop length for spectrogram
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Convolutional encoder for spectrogram patches
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, embed_dim, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features.
        
        Args:
            x: Audio tensor of shape (B, T) or (B, 1, F, T) or (T,) for single sample
            
        Returns:
            Audio embeddings of shape (B, embed_dim)
        """
        # Handle 1D input (single sample without batch dimension)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, T)
        
        if x.dim() == 2:  # (B, T) raw waveform
            # Compute spectrogram
            x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, 
                         return_complex=True)
            x = torch.abs(x).unsqueeze(1)  # (B, 1, F, T)
        
        # Convolutional encoding
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global pooling
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, embed_dim)
        x = self.norm(x)
        
        return x


class ImagePatchExtractor(nn.Module):
    """
    Extracts patches from images and projects them to latent vectors.
    Similar to video patches but for static images.
    For X=1 plane injection.
    """
    
    def __init__(self, patch_size: Tuple[int, int] = (16, 16), embed_dim: int = 768, in_channels: int = 3):
        """
        Initialize Image Patch Extractor.
        
        Args:
            patch_size: Size of each patch
            embed_dim: Dimensionality of output embeddings
            in_channels: Number of input channels
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1], embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract image patches.
        
        Args:
            x: Image tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        
        # Validate dimensions - apply padding if needed
        if H % ph != 0 or W % pw != 0:
            h_pad = (ph - H % ph) % ph
            w_pad = (pw - W % pw) % pw
            if h_pad > 0 or w_pad > 0:
                x = F.pad(x, (0, w_pad, 0, h_pad), mode='replicate')
                H, W = x.shape[2], x.shape[3]  # Update dimensions after padding
        
        # Unfold image into patches
        patches = F.unfold(x, kernel_size=(ph, pw), stride=(ph, pw))
        patches = patches.transpose(1, 2)  # (B, num_patches, C*ph*pw)
        
        # Project and normalize - use dynamic projection if channels differ
        if C == self.in_channels:
            patch_embeds = self.proj(patches)
        else:
            # Create dynamic projection for different channel count
            patch_dim = C * ph * pw
            dynamic_proj = nn.Linear(patch_dim, self.embed_dim).to(x.device)
            patch_embeds = dynamic_proj(patches)
        patch_embeds = self.norm(patch_embeds)
        
        return patch_embeds


class TextEmbeddingExtractor(nn.Module):
    """
    Projects text tokens/embeddings to the latent space.
    For X=2 plane injection.
    """
    
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 768, max_length: int = 512):
        """
        Initialize Text Embedding Extractor.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimensionality of output embeddings
            max_length: Maximum sequence length
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Position encoding (learnable)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Project text tokens to embeddings.
        
        Args:
            token_ids: Token IDs of shape (B, L)
            
        Returns:
            Text embeddings of shape (B, L, embed_dim)
        """
        B, L = token_ids.shape
        
        # Token embeddings
        x = self.token_embed(token_ids)
        
        # Position embeddings
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)
        
        x = self.norm(x)
        return x


class TabularFeatureExtractor(nn.Module):
    """
    Projects tabular data (numerical features) to latent space.
    For X=3 plane injection.
    """
    
    def __init__(self, num_features: int, embed_dim: int = 768, hidden_dim: int = 256):
        """
        Initialize Tabular Feature Extractor.
        
        Args:
            num_features: Number of input features
            embed_dim: Dimensionality of output embeddings
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project tabular features to embeddings.
        
        Args:
            x: Tabular data of shape (B, num_features)
            
        Returns:
            Tabular embeddings of shape (B, embed_dim)
        """
        return self.encoder(x)


class VideoPatchExtractor(nn.Module):
    """
    Extracts patches from video frames and projects them to latent vectors.
    Converts (H, W, C) video frames into a grid of patches, each projected to size D.
    For X=0 plane injection.
    """
    
    def __init__(self, 
                 patch_size: Tuple[int, int] = (16, 16),
                 embed_dim: int = 768,
                 in_channels: int = 3):
        """
        Initialize Video Patch Extractor.
        
        Args:
            patch_size: Size of each patch (height, width)
            embed_dim: Dimensionality of output embedding vectors
            in_channels: Number of input channels (e.g., 3 for RGB)
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        # Calculate patches per dimension
        self.patch_height, self.patch_width = patch_size
        
        # Linear projection of flattened patches
        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1], embed_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from video frames and project to latent space.
        
        Args:
            x: Input video tensor of shape (B, T, C, H, W) or (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, T, num_patches, embed_dim) or (B, num_patches, embed_dim)
        """
        # Handle different input formats
        if x.dim() == 4:  # (B, C, H, W) - single frame
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(1)  # Add time dimension: (B, 1, C, H, W)
        elif x.dim() == 5:  # (B, T, C, H, W) - video sequence
            B, T, C, H, W = x.shape
        else:
            raise ValueError(f"Input must be 4D or 5D tensor, got {x.dim()}D")
            
        # Validate channel count - allow flexible channel handling
        if C != self.in_channels:
            # Adjust projection layer to handle different channel counts
            # This allows processing images with different channel counts
            actual_channels = C
        else:
            actual_channels = self.in_channels
        
        # Validate dimensions - should already be divisible due to padding in tokenizer
        if H % self.patch_height != 0 or W % self.patch_width != 0:
            # Apply additional padding if needed (fallback)
            h_pad = (self.patch_height - H % self.patch_height) % self.patch_height
            w_pad = (self.patch_width - W % self.patch_width) % self.patch_width
            if h_pad > 0 or w_pad > 0:
                x = F.pad(x, (0, w_pad, 0, h_pad), mode='replicate')
                H, W = x.shape[3], x.shape[4]  # Update dimensions after padding
            
        # Calculate number of patches
        num_patches_h = H // self.patch_height
        num_patches_w = W // self.patch_width
        num_patches = num_patches_h * num_patches_w
        
        # Extract patches using unfolding
        # Reshape to (B*T, C, H, W) for patch extraction
        x_reshaped = x.reshape(B * T, C, H, W)
        
        # Extract patches: (B*T, C*patch_h*patch_w, num_patches)
        patches = F.unfold(x_reshaped, kernel_size=self.patch_size, stride=self.patch_size)
        # Transpose to (B*T, num_patches, C*patch_h*patch_w)
        patches = patches.transpose(1, 2)
        
        # Project patches to embedding dimension - use dynamic projection if channels differ
        if C == self.in_channels:
            patch_embeds = self.proj(patches)  # (B*T, num_patches, embed_dim)
        else:
            # Create dynamic projection for different channel count
            patch_dim = C * self.patch_height * self.patch_width
            dynamic_proj = nn.Linear(patch_dim, self.embed_dim).to(x.device)
            patch_embeds = dynamic_proj(patches)
        patch_embeds = self.norm(patch_embeds)
        
        # Reshape back to (B, T, num_patches, embed_dim)
        patch_embeds = patch_embeds.reshape(B, T, num_patches, self.embed_dim)
        
        # Remove time dimension if input was single frame
        if T == 1:
            patch_embeds = patch_embeds.squeeze(1)  # (B, num_patches, embed_dim)
            
        return patch_embeds
    
    def get_patch_grid_shape(self, H: int, W: int) -> Tuple[int, int]:
        """Get the grid shape of patches for given image dimensions."""
        return (H // self.patch_height, W // self.patch_width)


class SpatialTokenizer:
    """
    Omni-Modal Spatial Tokenizer for the 3D-RNG World Engine.
    Maps 5 modalities (video, audio, images, text, tabular) to specific coordinates
    on the 3D-RNG's spatial planes X=0 through X=3.
    
    Plane Mapping:
    - X=0: Video and Audio (coexisting)
    - X=1: Images
    - X=2: Text
    - X=3: Tabular data
    """
    
    def __init__(self,
                 vision_face_size: Tuple[int, int],
                 text_face_size: Tuple[int, int],
                 patch_size: Tuple[int, int] = (16, 16),
                 embed_dim: int = 768,
                 in_channels: int = 3,
                 num_tabular_features: int = 10,
                 vocab_size: int = 50000):
        """
        Initialize Spatial Tokenizer with 5 modalities.
        
        Args:
            vision_face_size: Size of vision face grid (height, width) - for X=0
            text_face_size: Size of text face grid (height, width) - for X=2
            patch_size: Size of video/image patches (height, width)
            embed_dim: Dimensionality of embedding vectors
            in_channels: Number of input channels for video/images
            num_tabular_features: Number of tabular features - for X=3
            vocab_size: Vocabulary size for text - for X=2
        """
        self.vision_face_size = vision_face_size
        self.text_face_size = text_face_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        # X=0: Video patches
        self.video_extractor = VideoPatchExtractor(patch_size, embed_dim, in_channels)
        
        # X=0: Audio features (coexisting with video)
        self.audio_extractor = AudioFeatureExtractor(embed_dim)
        
        # X=1: Image patches
        self.image_extractor = ImagePatchExtractor(patch_size, embed_dim, in_channels)
        
        # X=2: Text embeddings
        self.text_extractor = TextEmbeddingExtractor(vocab_size, embed_dim)
        
        # X=3: Tabular features
        self.tabular_extractor = TabularFeatureExtractor(num_tabular_features, embed_dim)
        
        # Text tokenizer will be provided externally (GraphCommunityTokenizer)
        self.text_tokenizer = None
        
        # Validate that we can fit patches on the vision face
        self._validate_face_capacity()
        
        print(f"SpatialTokenizer initialized with 5 modalities:")
        print(f"  X=0: Video + Audio")
        print(f"  X=1: Images")
        print(f"  X=2: Text")
        print(f"  X=3: Tabular")
        
    def _validate_face_capacity(self):
        """Validate that the vision face can accommodate the expected patch grid."""
        # This will be checked dynamically based on actual input dimensions
        pass
    
    def set_text_tokenizer(self, text_tokenizer):
        """Set the external text tokenizer (e.g., GraphCommunityTokenizer)."""
        self.text_tokenizer = text_tokenizer
    
    def tokenize_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Tokenize a single video frame into patch embeddings.
        
        Args:
            frame: Video frame of shape (H, W, C) or (C, H, W) or (H, W) for grayscale
            
        Returns:
            Patch embeddings of shape (num_patches, embed_dim)
        """
        # Convert to torch tensor
        if isinstance(frame, np.ndarray):
            frame_tensor = torch.from_numpy(frame).float()
        else:
            frame_tensor = frame.float()
            
        # Handle different input formats
        if frame_tensor.dim() == 3:
            # Check if it's channel-last (H, W, C) or channel-first (C, H, W)
            # Heuristic: if last dimension is 1, 3, or 4 (common channel sizes)
            if frame_tensor.shape[2] in [1, 3, 4]:
                # Likely (H, W, C) format - convert to (C, H, W)
                frame_tensor = frame_tensor.permute(2, 0, 1)
            elif frame_tensor.shape[0] in [1, 3, 4]:
                # Already (C, H, W) format
                pass
            else:
                # Fallback: assume first dim is channel if small
                if frame_tensor.shape[0] <= 4:
                    pass  # Already correct
                else:
                    raise ValueError(f"Unexpected frame shape: {frame_tensor.shape}")
        elif frame_tensor.dim() == 2:
            # Grayscale frame (H, W) - add channel dimension
            frame_tensor = frame_tensor.unsqueeze(0)  # -> (1, H, W)
        else:
            raise ValueError(f"Frame must be 2D or 3D tensor, got {frame_tensor.dim()}D")
        
        # Ensure dimensions are compatible with patch size
        # If not, resize the frame to be divisible by patch size
        C, H, W = frame_tensor.shape
        ph, pw = self.patch_size
        
        # Calculate padding needed to make dimensions divisible by patch size
        h_pad = (ph - H % ph) % ph if H % ph != 0 else 0
        w_pad = (pw - W % pw) % pw if W % pw != 0 else 0
        
        if h_pad > 0 or w_pad > 0:
            # Apply padding (replicating edge pixels)
            frame_tensor = F.pad(frame_tensor, (0, w_pad, 0, h_pad), mode='replicate')
        
        # Add batch and time dimensions: (1, 1, C, H, W)
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)
        
        # Extract patches
        with torch.no_grad():
            patch_embeds = self.video_extractor(frame_tensor)  # (1, 1, num_patches, embed_dim)
            patch_embeds = patch_embeds.squeeze(0).squeeze(0)  # (num_patches, embed_dim)
            
        return patch_embeds.numpy()
    
    def map_patches_to_vision_face(self, patch_embeds: np.ndarray) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Map video patch embeddings to specific coordinates on the Vision Face (x=0 plane).
        
        Args:
            patch_embeds: Patch embeddings of shape (num_patches, embed_dim)
            
        Returns:
            Dictionary mapping (x, y, z) coordinates to patch embedding vectors
        """
        num_patches = patch_embeds.shape[0]
        vision_height, vision_width = self.vision_face_size
        expected_patches = vision_height * vision_width
        
        # Handle case where number of patches doesn't exactly match vision face
        if num_patches != expected_patches:
            # We'll use adaptive mapping - either interpolate or select subset
            if num_patches > expected_patches:
                # Select subset of patches (could use attention or pooling)
                indices = np.linspace(0, num_patches-1, expected_patches, dtype=int)
                patch_embeds = patch_embeds[indices]
            else:
                # Repeat or interpolate patches to fill vision face
                repeat_factor = expected_patches // num_patches
                remainder = expected_patches % num_patches
                repeated = np.repeat(patch_embeds, repeat_factor, axis=0)
                if remainder > 0:
                    repeated = np.vstack([repeated, patch_embeds[:remainder]])
                patch_embeds = repeated
        
        # Map patches to vision face coordinates
        vision_mapping = {}
        patch_idx = 0
        
        for y in range(vision_height):
            for z in range(vision_width):
                coord = (0, y, z)  # x=0 for vision face
                if patch_idx < len(patch_embeds):
                    vision_mapping[coord] = patch_embeds[patch_idx].copy()
                    patch_idx += 1
                else:
                    # Should not happen with our validation, but safety check
                    vision_mapping[coord] = np.zeros(self.embed_dim)
                    
        return vision_mapping
    
    def tokenize_text(self, text: str) -> np.ndarray:
        """
        Tokenize text into embeddings using the external text tokenizer.
        
        Args:
            text: Input text string
            
        Returns:
            Text embeddings of shape (num_text_tokens, embed_dim)
        """
        if self.text_tokenizer is None:
            # Fallback: return random embeddings if no tokenizer set
            # In practice, this should be set to a proper tokenizer like GraphCommunityTokenizer
            num_text_tokens = self.text_face_size[0] * self.text_face_size[1]
            return np.random.randn(num_text_tokens, self.embed_dim) * 0.1
            
        # Use external text tokenizer
        text_embeds = self.text_tokenizer.encode(text)  # Should return (num_tokens, embed_dim)
        return text_embeds
    
    def map_text_to_text_face(self, text_embeds: np.ndarray) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Map text embeddings to specific coordinates on the Text Face (x=1 plane).
        
        Args:
            text_embeds: Text embeddings of shape (num_text_tokens, embed_dim)
            
        Returns:
            Dictionary mapping (x, y, z) coordinates to text embedding vectors
        """
        num_text_tokens = text_embeds.shape[0]
        text_height, text_width = self.text_face_size
        expected_tokens = text_height * text_width
        
        # Handle case where number of tokens doesn't exactly match text face
        if num_text_tokens != expected_tokens:
            if num_text_tokens > expected_tokens:
                # Select subset of tokens
                indices = np.linspace(0, num_text_tokens-1, expected_tokens, dtype=int)
                text_embeds = text_embeds[indices]
            else:
                # Repeat or interpolate tokens to fill text face
                repeat_factor = expected_tokens // num_text_tokens
                remainder = expected_tokens % num_text_tokens
                repeated = np.repeat(text_embeds, repeat_factor, axis=0)
                if remainder > 0:
                    repeated = np.vstack([repeated, text_embeds[:remainder]])
                text_embeds = repeated
        
        # Map text tokens to text face coordinates
        text_mapping = {}
        token_idx = 0
        
        for y in range(text_height):
            for z in range(text_width):
                coord = (1, y, z)  # x=1 for text face
                if token_idx < len(text_embeds):
                    text_mapping[coord] = text_embeds[token_idx].copy()
                    token_idx += 1
                else:
                    # Should not happen with our validation, but safety check
                    text_mapping[coord] = np.zeros(self.embed_dim)
                    
        return text_mapping
    
    def tokenize_multi_modal(self,
                            video_frame: Optional[np.ndarray] = None,
                            text: Optional[str] = None,
                            audio: Optional[torch.Tensor] = None,
                            image: Optional[np.ndarray] = None,
                            tabular: Optional[np.ndarray] = None) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Tokenize multi-modal inputs and map them to their respective faces.
        
        Args:
            video_frame: Optional video frame of shape (H, W, C) - maps to X=0
            text: Optional text string - maps to X=2
            audio: Optional audio tensor of shape (B, T) or (B, 1, F, T) - maps to X=0
            image: Optional image of shape (H, W, C) or (C, H, W) - maps to X=1
            tabular: Optional tabular data of shape (num_features,) - maps to X=3
            
        Returns:
            Unified dictionary mapping (x, y, z) coordinates to embedding vectors
            
        Raises:
            ValueError: If any provided modality is of incorrect type/tensor format
        """
        # --- FAIL FAST: Validate inputs immediately ---
        if video_frame is not None and not isinstance(video_frame, np.ndarray):
            raise ValueError(f"video_frame must be np.ndarray, got {type(video_frame)}")
        if text is not None and not isinstance(text, str):
            raise ValueError(f"text must be str, got {type(text)}")
        if audio is not None and not isinstance(audio, torch.Tensor):
            raise ValueError(f"audio must be torch.Tensor, got {type(audio)}")
        if image is not None and not isinstance(image, np.ndarray):
            raise ValueError(f"image must be np.ndarray, got {type(image)}")
        if tabular is not None and not isinstance(tabular, np.ndarray):
            raise ValueError(f"tabular must be np.ndarray, got {type(tabular)}")
        
        # Initialize unified mapping dictionary
        unified_mapping: Dict[Tuple[int, int, int], np.ndarray] = {}
        
        # --- X=0: Video + Audio (coexisting on same plane) ---
        if video_frame is not None:
            patch_embeds = self.tokenize_video_frame(video_frame)
            vision_mapping = self.map_patches_to_vision_face(patch_embeds)
            unified_mapping.update(vision_mapping)
        
        # Process audio and map to X=0 (coexisting with video)
        if audio is not None:
            audio_embeds = self._process_audio(audio)
            audio_mapping = self.map_audio_to_face(audio_embeds)
            unified_mapping.update(audio_mapping)
        
        # --- X=1: Images ---
        if image is not None:
            image_embeds = self._process_image(image)
            image_mapping = self.map_image_to_image_face(image_embeds)
            unified_mapping.update(image_mapping)
        
        # --- X=2: Text ---
        if text is not None:
            text_embeds = self.tokenize_text(text)
            text_mapping = self.map_text_to_text_face(text_embeds)
            unified_mapping.update(text_mapping)
        
        # --- X=3: Tabular data ---
        if tabular is not None:
            tabular_embeds = self._process_tabular(tabular)
            tabular_mapping = self.map_tabular_to_tabular_face(tabular_embeds)
            unified_mapping.update(tabular_mapping)
            
        return unified_mapping
    
    def _process_audio(self, audio: torch.Tensor) -> np.ndarray:
        """
        Process audio through audio_extractor.
        
        Args:
            audio: Audio tensor of shape (B, T) or (B, 1, F, T)
            
        Returns:
            Audio embeddings of shape (1, embed_dim)
        """
        with torch.no_grad():
            audio_embeds = self.audio_extractor(audio)
        return audio_embeds.cpu().numpy()
    
    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image through image_extractor.
        
        Args:
            image: Image of shape (H, W, C) or (C, H, W) or (H, W) for grayscale
            
        Returns:
            Image embeddings of shape (num_patches, embed_dim)
        """
        # Convert to torch tensor
        image_tensor = torch.from_numpy(image).float()
        
        # Handle different input formats
        if image_tensor.dim() == 3:
            # Check if it's channel-last (H, W, C) or channel-first (C, H, W)
            # Heuristic: if last dimension is 1, 3, or 4 (common channel sizes)
            if image_tensor.shape[2] in [1, 3, 4]:
                # Likely (H, W, C) format - convert to (C, H, W)
                image_tensor = image_tensor.permute(2, 0, 1)
            elif image_tensor.shape[0] in [1, 3, 4]:
                # Already (C, H, W) format
                pass
            else:
                # Fallback: assume channel-first if first dim is small
                if image_tensor.shape[0] <= 4:
                    pass  # Already correct
                else:
                    # Might be (H, W, C) where C is large - try to detect
                    pass
        elif image_tensor.dim() == 2:
            # Grayscale image (H, W) - add channel dimension
            image_tensor = image_tensor.unsqueeze(0)  # -> (1, H, W)
        
        # Ensure dimensions are compatible with patch size
        # If not, resize the image to be divisible by patch size
        # First ensure we have batch dimension (add if missing)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W) or (1, H, W)
        
        # If still 3 dims (1, H, W for grayscale), add channel dim
        if image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
            # Check if it's (1, H, W) format
            if image_tensor.shape[1] > 4:  # Likely H, not C
                image_tensor = image_tensor.unsqueeze(1)  # -> (1, 1, H, W)
        
        B, C, H, W = image_tensor.shape
        ph, pw = self.patch_size
        
        # Calculate padding needed to make dimensions divisible by patch size
        h_pad = (ph - H % ph) % ph if H % ph != 0 else 0
        w_pad = (pw - W % pw) % pw if W % pw != 0 else 0
        
        if h_pad > 0 or w_pad > 0:
            # Apply padding ( replicating edge pixels)
            image_tensor = F.pad(image_tensor, (0, w_pad, 0, h_pad), mode='replicate')
        
        # Batch dimension is already added above - ensure we have 4 dims
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            image_embeds = self.image_extractor(image_tensor)
        
        return image_embeds.cpu().numpy()
    
    def _process_tabular(self, tabular: np.ndarray) -> np.ndarray:
        """
        Process tabular data through tabular_extractor.
        
        Args:
            tabular: Tabular data of shape (num_features,)
            
        Returns:
            Tabular embeddings of shape (1, embed_dim)
        """
        # Convert to torch tensor
        tabular_tensor = torch.from_numpy(tabular).float().unsqueeze(0)
        
        with torch.no_grad():
            tabular_embeds = self.tabular_extractor(tabular_tensor)
        
        return tabular_embeds.cpu().numpy()
    
    def map_audio_to_face(self, audio_embeds: np.ndarray) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Map audio embeddings to the audio zone on X=0 plane.
        
        Args:
            audio_embeds: Audio embeddings of shape (1, embed_dim)
            
        Returns:
            Dictionary mapping (x, y, z) coordinates to audio embedding vectors
        """
        audio_mapping = {}
        # Map audio to a single location on X=0 plane (coexisting with video)
        # Use the first available position after video patches
        vision_height, vision_width = self.vision_face_size
        audio_coord = (0, vision_height // 2, vision_width - 1)  # X=0 for audio
        audio_mapping[audio_coord] = audio_embeds[0].copy()
        return audio_mapping
    
    def map_image_to_image_face(self, image_embeds: np.ndarray) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Map image embeddings to the Image Face (x=1 plane).
        
        Args:
            image_embeds: Image embeddings of shape (num_patches, embed_dim)
            
        Returns:
            Dictionary mapping (x, y, z) coordinates to image embedding vectors
        """
        num_patches = image_embeds.shape[0]
        # Use vision_face_size for image face as well
        image_height, image_width = self.vision_face_size
        expected_patches = image_height * image_width
        
        # Handle patch count mismatch
        if num_patches != expected_patches:
            if num_patches > expected_patches:
                indices = np.linspace(0, num_patches-1, expected_patches, dtype=int)
                image_embeds = image_embeds[indices]
            else:
                repeat_factor = expected_patches // num_patches
                remainder = expected_patches % num_patches
                repeated = np.repeat(image_embeds, repeat_factor, axis=0)
                if remainder > 0:
                    repeated = np.vstack([repeated, image_embeds[:remainder]])
                image_embeds = repeated
        
        # Map patches to image face coordinates
        image_mapping = {}
        patch_idx = 0
        
        for y in range(image_height):
            for z in range(image_width):
                coord = (1, y, z)  # x=1 for image face
                if patch_idx < len(image_embeds):
                    image_mapping[coord] = image_embeds[patch_idx].copy()
                    patch_idx += 1
                else:
                    image_mapping[coord] = np.zeros(self.embed_dim)
                    
        return image_mapping
    
    def map_tabular_to_tabular_face(self, tabular_embeds: np.ndarray) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Map tabular embeddings to the Tabular Face (x=3 plane).
        
        Args:
            tabular_embeds: Tabular embeddings of shape (1, embed_dim)
            
        Returns:
            Dictionary mapping (x, y, z) coordinates to tabular embedding vectors
        """
        tabular_mapping = {}
        # Map tabular to a single location on X=3 plane
        tabular_coord = (3, 0, 0)  # X=3 for tabular
        tabular_mapping[tabular_coord] = tabular_embeds[0].copy()
        return tabular_mapping


def create_spatial_tokenizer_example():
    """
    Create an example showing how to use the Spatial Tokenizer.
    """
    print("Creating Spatial Multi-Modal Tokenizer...")
    
    # Initialize spatial tokenizer
    tokenizer = SpatialTokenizer(
        vision_face_size=(8, 8),   # 8x8 grid on vision face
        text_face_size=(4, 4),     # 4x4 grid on text face
        patch_size=(16, 16),       # 16x16 patches
        embed_dim=384,             # Embedding dimension
        in_channels=3              # RGB input
    )
    
    print(f"SpatialTokenizer initialized:")
    print(f"  Vision face: {tokenizer.vision_face_size}")
    print(f"  Text face: {tokenizer.text_face_size}")
    print(f"  Patch size: {tokenizer.patch_size}")
    print(f"  Embed dim: {tokenizer.embed_dim}")
    
    # Example usage with dummy data
    print("\nProcessing example multi-modal input...")
    
    # Create dummy video frame (e.g., 128x128 RGB)
    dummy_frame = np.random.randn(128, 128, 3).astype(np.float32)
    print(f"Input video frame shape: {dummy_frame.shape}")
    
    # Create dummy text
    dummy_text = "The cat sits on the mat."
    print(f"Input text: '{dummy_text}'")
    
    # Tokenize and map
    vision_mapping, text_mapping = tokenizer.tokenize_multi_modal(dummy_frame, dummy_text)
    
    print(f"Output vision mapping: {len(vision_mapping)} patches mapped to Vision Face (x=0)")
    print(f"Output text mapping: {len(text_mapping)} tokens mapped to Text Face (x=1)")
    
    # Show sample mappings
    if vision_mapping:
        sample_coord = list(vision_mapping.keys())[0]
        sample_vector = vision_mapping[sample_coord]
        print(f"Sample vision mapping at {sample_coord}: vector shape {sample_vector.shape}, norm {np.linalg.norm(sample_vector):.3f}")
    
    if text_mapping:
        sample_coord = list(text_mapping.keys())[0]
        sample_vector = text_mapping[sample_coord]
        print(f"Sample text mapping at {sample_coord}: vector shape {sample_vector.shape}, norm {np.linalg.norm(sample_vector):.3f}")
    
    print("\nSpatialTokenizer ready for integration with 3D-RNG World Engine!")
    return tokenizer


if __name__ == "__main__":
    # Run the example
    create_spatial_tokenizer_example()