import argparse
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from recognition_training import read_image
import faiss
import pickle
import numpy as np

def visualize(args):
    data_dir_gallery = Path(args.data_dir_gallery, '**', '**', '*.jpg')
    data_dir_test = Path(args.data_dir_test, '**', '**', '*.jpg')
    target_shape = (args.target_shape, args.target_shape)
    
    # 1. Load gallery paths
    list_ds_gallery = tf.data.Dataset.list_files(str(data_dir_gallery), shuffle=False)
    gallery_paths = list(list_ds_gallery.as_numpy_iterator())
    label_list = [int(Path(p.decode()).parent.name) for p in gallery_paths]
    
    # 2. Load Gallery Embeddings
    print("Loading gallery embeddings...")
    with open('embeddings.pickle', 'rb') as f:
        embeddingTotale = pickle.load(f)
    
    # 3. Load model for test embeddings
    print("Loading model...")
    embeddingNet = tf.keras.models.load_model(args.model)
    
    # 4. Load test images
    list_ds_test = tf.data.Dataset.list_files(str(data_dir_test), shuffle=True)
    test_paths = list(list_ds_test.take(args.num_visualize).as_numpy_iterator())
    
    # Process test images
    datasetTest = tf.data.Dataset.from_tensor_slices(test_paths)
    datasetTest = datasetTest.map(lambda x: read_image(x, target_shape))
    datasetTest = datasetTest.batch(args.num_visualize)
    
    # 5. Predict Test Embeddings
    print("Extracting test embeddings...")
    embeddingTest = embeddingNet.predict(datasetTest)
    
    # 6. FAISS similarity search
    k = 5 # Top 5 Matches
    index = faiss.IndexFlatIP(args.embedding_size)
    faiss.normalize_L2(embeddingTotale)
    index.add(embeddingTotale)
    
    print("Searching for matches...")
    D, I = index.search(embeddingTest, k)
    
    # 7. Visualization
    fig, axes = plt.subplots(args.num_visualize, k + 1, figsize=(15, 3 * args.num_visualize))
    if args.num_visualize == 1:
        axes = [axes]
        
    for i in range(args.num_visualize):
        test_path = test_paths[i].decode()
        true_label = Path(test_path).parent.name
        
        # Original Image
        ax = axes[i][0]
        img = plt.imread(test_path)
        ax.imshow(img)
        ax.set_title(f"Query\nTrue: {true_label}", fontweight='bold')
        ax.axis('off')
        
        # Matches
        for j in range(k):
            match_idx = I[i][j]
            match_path = gallery_paths[match_idx].decode()
            match_label = label_list[match_idx]
            match_dist = D[i][j]
            
            ax_m = axes[i][j+1]
            try:
                img_m = plt.imread(match_path)
                ax_m.imshow(img_m)
            except Exception:
                pass
            
            color = "green" if str(match_label) == str(true_label) else "red"
            ax_m.set_title(f"Match {j+1}\nPred: {match_label}\nSim: {match_dist:.2f}", color=color)
            ax_m.axis('off')
            
    plt.tight_layout()
    output_filename = 'visualization_results.png'
    plt.savefig(output_filename, dpi=150)
    print(f"Saved visualization to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_gallery', type=str, required=True)
    parser.add_argument('--data_dir_test', type=str, required=True)
    parser.add_argument('--target_shape', type=int, default=224)
    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num_visualize', type=int, default=5)
    args = parser.parse_args()
    visualize(args)
