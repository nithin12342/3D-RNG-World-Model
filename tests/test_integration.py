"""
Integration test for 3D-RNG components
Tests the integration of graph_tokenizer.py, parallel_graph_impl.py, and parallel_eval.py
"""

import numpy as np
import sys
import os

def test_graph_tokenizer():
    """Test the GraphCommunityTokenizer component."""
    print("Testing GraphCommunityTokenizer...")
    
    try:
        from core.graph_tokenizer import GraphCommunityTokenizer
        
        # Create tokenizer
        tokenizer = GraphCommunityTokenizer(min_char_freq=1, min_transition_prob=0.01)
        
        # Test with simple corpus
        corpus = [
            "hello world",
            "machine learning",
            "neural networks"
        ]
        
        # Train tokenizer
        tokenizer.fit(corpus)
        
        # Test encoding/decoding
        test_text = "hello"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"  Original: '{test_text}'")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
        
        assert isinstance(encoded, list), "Encoded output should be a list"
        assert all(isinstance(t, int) for t in encoded), "All tokens should be integers"
        assert isinstance(decoded, str), "Decoded output should be a string"
        
        print("  * GraphCommunityTokenizer test passed")
        return True
        
    except Exception as e:
        print(f"  ! GraphCommunityTokenizer test failed: {e}")
        return False

def test_parallel_graph():
    """Test the ParallelNeuralGraph3D component."""
    print("\nTesting ParallelNeuralGraph3D...")
    
    try:
        from core.parallel_graph_impl import ParallelNeuralGraph3D
        
        # Create graph
        graph = ParallelNeuralGraph3D(
            dim_x=6, dim_y=4, dim_z=4,
            hidden_size=8,
            input_face_size=(2, 2),  # 4 input channels
            output_face_size=(2, 2)  # 4 output channels
        )
        
        # Test forward probe
        input_data = np.random.randn(4, 8) * 0.5  # 4 chunks, 8-dim each
        output_tensor, paths = graph.forward_probe(input_data, max_steps=20)
        
        print(f"  Input shape: {input_data.shape}")
        print(f"  Output shape: {output_tensor.shape}")
        print(f"  Number of paths: {len(paths)}")
        if paths:
            print(f"  Average path length: {np.mean([len(p) for p in paths]):.1f}")
        
        # Test traceback reinforcement
        if len(paths) > 0:
            rewards = np.random.rand(len(paths)) * 2 - 1  # Random rewards [-1, 1]
            graph.traceback_reinforcement(paths, rewards)
            print(f"  Applied reinforcement with rewards: {rewards}")
        
        # Test statistics
        stats = graph.get_activation_statistics()
        face_stats = graph.get_face_activation_stats()
        
        print(f"  Activation ratio: {stats['activation_ratio']:.2%}")
        print(f"  Input face active: {face_stats['input_face']['ratio']:.2%}")
        print(f"  Output face active: {face_stats['output_face']['ratio']:.2%}")
        
        assert output_tensor.shape[0] <= 4, "Output should not exceed output face size"
        assert isinstance(paths, list), "Paths should be a list"
        
        print("  * ParallelNeuralGraph3D test passed")
        return True
        
    except Exception as e:
        print(f"  ! ParallelNeuralGraph3D test failed: {e}")
        return False

def test_evaluator_setup():
    """Test the ParallelRNGEvaluator setup."""
    print("\nTesting ParallelRNGEvaluator setup...")
    
    try:
        from core.parallel_eval import ParallelRNGEvaluator
        
        # Create evaluator
        evaluator = ParallelRNGEvaluator(
            graph_dimensions=(6, 4, 4),
            hidden_size=8,
            input_face_size=(2, 2),
            output_face_size=(2, 2)
        )
        
        # Test that components are None initially
        assert evaluator.graph is None, "Graph should be None initially"
        assert evaluator.tokenizer is None, "Tokenizer should be None initially"
        
        # Test setup_components
        evaluator.setup_components()
        
        # Test that components are now initialized
        assert evaluator.graph is not None, "Graph should be initialized"
        assert evaluator.tokenizer is not None, "Tokenizer should be initialized"
        
        print("  * ParallelRNGEvaluator setup test passed")
        return True
        
    except Exception as e:
        print(f"  ! ParallelRNGEvaluator setup test failed: {e}")
        return False

def test_text_processing():
    """Test text processing pipeline."""
    print("\nTesting text processing pipeline...")
    
    try:
        from core.parallel_eval import ParallelRNGEvaluator
        from core.graph_tokenizer import GraphCommunityTokenizer
        
        # Create evaluator (which will create tokenizer)
        evaluator = ParallelRNGEvaluator(
            graph_dimensions=(6, 4, 4),
            hidden_size=8,
            input_face_size=(2, 2),
            output_face_size=(2, 2)
        )
        evaluator.setup_components()
        
        # Test text to chunks
        text = "hello world this is a test"
        chunks = evaluator.text_to_chunks(text, chunk_size=4)
        
        print(f"  Original text: '{text}'")
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  Chunk sizes: {[len(c) for c in chunks]}")
        
        # Test chunks to tensor
        if len(chunks) > 0:
            input_tensor = evaluator.chunks_to_input_tensor(chunks)
            print(f"  Input tensor shape: {input_tensor.shape}")
            
            assert input_tensor.shape[0] == len(chunks), "First dimension should match chunk count"
            assert input_tensor.shape[1] == evaluator.hidden_size, "Second dimension should match hidden size"
        
        print("  * Text processing pipeline test passed")
        return True
        
    except Exception as e:
        print(f"  ! Text processing pipeline test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("3D-RNG COMPONENT INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        test_graph_tokenizer,
        test_parallel_graph,
        test_evaluator_setup,
        test_text_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print(":) All integration tests passed!")
        return 0
    else:
        print(":( Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())