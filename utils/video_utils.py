import av
import numpy as np
import cv2


def save3d(output_path, list_of_imgs, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height, _ = list_of_imgs[0].shape
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in list_of_imgs:
        video_writer.write(frame)
    video_writer.release()


def video_heatmap_colors(heats):
    outputs = []
    for heat in heats:
        # Normalize the data to the range [0, 255]
        normalized_data = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
        # Apply the color map (heatmap)
        heatmap = cv2.applyColorMap(np.uint8(normalized_data), cv2.COLORMAP_JET)
        outputs.append(heatmap)
    return np.stack(outputs)


def video_alpha_blending(heats, ori_imgs, frame_size):
    blended = []
    for ih, ori_img in enumerate(ori_imgs):
        img = cv2.resize(ori_img, frame_size)
        heat = heats[:, :, ih]
        normalized_data = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
        hm = cv2.applyColorMap(np.uint8(normalized_data), cv2.COLORMAP_JET)
        combined_image = cv2.addWeighted(img, 0.5, hm, 0.5, 0)
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        blended.append(combined_image)
    return np.stack(blended)


def read_video_pyav(file_path, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    container = av.open(file_path)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices