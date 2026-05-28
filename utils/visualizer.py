import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, morphology
from skimage.morphology import remove_small_objects

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def export_hist(scores, save_path, title=''):
    plt.figure()
    plt.hist(scores[~np.isnan(scores)], bins=50)
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def export_groundtruth(dataset, save_path):
    for i in range(len(dataset)):
        img, label, mask, file_name, _ = dataset[i]
        img = denormalization(img)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img)
        ax[0].set_title('Image')
        ax[0].axis('off')
        
        ax[1].imshow(mask[0], cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')
        
        os.makedirs(os.path.join(save_path, 'ground_truth'), exist_ok=True)
        plt.savefig(os.path.join(save_path, 'ground_truth', f'{file_name}.png'), bbox_inches='tight')
        plt.close()

def export_scores(scores, save_path, title=''):
    """Export score map"""
    plt.figure()
    plt.imshow(scores, cmap='jet')
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def export_test_images(image_list, scores, gt_mask_list, save_dir, file_names):
    """Export test image results"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(image_list)):
        image = denormalization(image_list[i])
        score_map = scores[i]
        gt_mask = gt_mask_list[i].squeeze()
        
        # generate high-quality mask
        pred_mask = generate_high_quality_mask(score_map)
        pred_edge = generate_precise_edges(pred_mask)
        
        # create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # anomaly score map
        im1 = axes[1].imshow(score_map, cmap='jet')
        axes[1].set_title('Anomaly Score Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # predicted defect mask
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        # ground truth mask
        axes[3].imshow(gt_mask, cmap='gray')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{file_names[i]}_result.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def generate_high_quality_mask(scores, threshold=0.5, min_size=20):
    """Generate high-quality defect mask"""
    # threshold -> binarization
    mask = (scores > threshold)

    mask = remove_small_objects(mask.astype(bool), min_size=min_size)

    return (mask.astype(np.uint8) * 255)

def generate_precise_edges(mask):
    """Generate precise defect edges"""
    # Canny edge detection
    edges = feature.canny(mask.astype(bool), sigma=1)
    
    # morphological thinning
    edges = morphology.skeletonize(edges)
    
    return edges.astype(np.uint8) * 255

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """Overlay mask on image"""
    image = image.copy()
    mask = mask.astype(bool)
    image[mask] = image[mask] * (1 - alpha) + np.array(color) * alpha
    return image.astype(np.uint8)

def extract_filename_info(file_path, img_type=None):

    # handle different input types (list/tuple/pathlib)
    if isinstance(file_path, (list, tuple)):
        if len(file_path) == 0:
            return "unknown"
        file_path = file_path[0]

    # convert to string and normalize separators
    file_str = str(file_path)
    file_str = file_str.replace('\\', '/')

    # extract filename without extension
    filename = os.path.splitext(os.path.basename(file_str))[0]

    # if input contains path separator, try using parent dir name as subclass name
    if '/' in file_str:
        parent_dir = os.path.basename(os.path.dirname(file_str))
        if parent_dir and parent_dir not in ['.', '']:
            return f"{parent_dir}_{filename}"

    # if no path info, use img_type (if provided) as subclass name
    if img_type and isinstance(img_type, str):
        return f"{img_type}_{filename}"

    # fallback: return filename only
    return filename

def plot_visualizing_results(image_list, scores, img_scores, gt_mask_list, pix_threshold, img_threshold, save_dir, file_names, img_types):
    """Improved visualization function with more meaningful filenames"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(image_list)):
        image = denormalization(image_list[i])
        score_map = scores[i]
        img_score = img_scores[i]
        gt_mask = gt_mask_list[i].squeeze()
        
        # generate high-quality mask
        pred_mask = generate_high_quality_mask(score_map, threshold=pix_threshold)
        pred_edge = generate_precise_edges(pred_mask)
        
        # overlay mask on image
        overlay = overlay_mask(image, pred_mask)
        
        # get current file info
        file_name = file_names[i] if i < len(file_names) else f"image_{i}"
        img_type = img_types[i] if i < len(img_types) else "unknown"
        
        # extract meaningful filename info
        base_name = extract_filename_info(file_name, img_type)
        # print(f"Processing {base_name} (img_type: {img_type})...")
        
        # create visualization
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        # original image
        axes[0].imshow(image)
        axes[0].set_title(f'Original ({img_type})\nScore: {img_score:.2f}')
        axes[0].axis('off')
        
        # anomaly score map
        im1 = axes[1].imshow(score_map, cmap='jet')
        axes[1].set_title('Anomaly Score Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # predicted defect mask
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        # predicted defect edges
        axes[3].imshow(pred_edge, cmap='gray')
        axes[3].set_title('Defect Edges')
        axes[3].axis('off')
        
        # overlay image
        axes[4].imshow(overlay)
        axes[4].set_title('Overlay')
        axes[4].axis('off')
        
        plt.tight_layout()
        
        # save combined result image
        result_path = os.path.join(save_dir, f'{base_name}_result.png')
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        plt.close()
        # print(f"Saved result to: {result_path}")
        
        # save heat map / mask / edge / overlay separately
        heat_map_path = os.path.join(save_dir, f'{base_name}_heat_map.png')
        mask_path = os.path.join(save_dir, f'{base_name}_mask.png')
        edge_path = os.path.join(save_dir, f'{base_name}_edge.png')
        overlay_path = os.path.join(save_dir, f'{base_name}_overlay.png')

        if np.max(score_map) > np.min(score_map):
            heat_map_to_save = (score_map - np.min(score_map)) / (np.max(score_map) - np.min(score_map))
        else:
            heat_map_to_save = np.zeros_like(score_map)
        
        plt.imsave(heat_map_path, heat_map_to_save, cmap='jet', vmin=0.0, vmax=1.0)
        plt.imsave(mask_path, pred_mask, cmap='gray')
        plt.imsave(edge_path, pred_edge, cmap='gray')
        plt.imsave(overlay_path, overlay)  # save overlay image directly
        
        # print(f"Saved mask to: {mask_path}")
        # print(f"Saved edge to: {edge_path}")
        # print(f"Saved overlay to: {overlay_path}")
