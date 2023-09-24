import cv2
import numpy as np

# Create a 10x10 image with a specified character in the center
def create_char_image(char):
    image = np.ones((16, 12, 3), dtype=np.uint8) * 255  # 3-channel image (RGB) initialized with zeros
    size, _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x_offset = (16 - size[0]) // 2
    y_offset = (16 + size[1]) // 2
    cv2.putText(image, char, (x_offset-2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def create_word_image(word):
    word = word.upper()
    # Create images with characters 'A', 'P', 'P', 'L', and 'E'
    char_images = {c: create_char_image(c) for c in set(word)}
    # Combine the images horizontally to form "APPLE"
    word_image = np.hstack([char_images[c] for c in word])
    return word_image


def create_words_image(words, heats=None):
    heats = [[] for _ in words] if heats is None else heats
    heat_intensities = []
    word_imgs = []
    for word, heat in zip(words, heats):
        w_img = create_word_image(word)
        w_img = np.hstack([w_img, np.ones((16, 10, 3), dtype=np.uint8) * 255])
        word_imgs.append(w_img)
        if heat is not None:
            heat_vals = np.ones(w_img.shape[:2]) * heat
            heat_intensities.append(heat_vals)
    heat_intensities = (np.hstack(heat_intensities) * 255).astype(np.uint8)
    word_imgs = np.hstack(word_imgs)
    if heats is not None:
        heat_intensities = cv2.applyColorMap(heat_intensities, cv2.COLORMAP_JET)
        #heat_intensities = colored_heats(heat_intensities)
        heat_intensities = cv2.GaussianBlur(heat_intensities, (23, 13), 21)
        word_imgs = cv2.addWeighted(heat_intensities, 0.5, word_imgs, 0.5, 0)

    return word_imgs


def vconcat_images_max_width(images):
    max_width = max(image.shape[1] for image in images)
    concatenated_image = np.zeros((0, max_width, 3), dtype=np.uint8)

    for image in images:
        padding = max_width - image.shape[1]
        left_padding = padding // 2
        right_padding = padding - left_padding
        padded_image = cv2.copyMakeBorder(image, 0, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        concatenated_image = np.vstack((concatenated_image, padded_image))

    return concatenated_image


MAX_VIZ_TOKENS = 10


def text_class_activation_map(tokens, importance, start_end_ids=None):
    start_id, end_id = (0, len(tokens)) if start_end_ids is None else start_end_ids
    viz_toks, viz_imps = tokens[start_id:end_id], importance[start_id:end_id]
    tok_cam_viz_imgs = []
    for i in range(0, len(viz_toks), MAX_VIZ_TOKENS):
        cam_viz_img = create_words_image(viz_toks[i:i+MAX_VIZ_TOKENS], viz_imps[i:i+MAX_VIZ_TOKENS])
        tok_cam_viz_imgs.append(cam_viz_img)

    tok_cam_viz_imgs = vconcat_images_max_width(tok_cam_viz_imgs)
    return tok_cam_viz_imgs


if __name__ == '__main__':

    img = create_word_image('APPLE')
    cv2.imwrite('ig_data/test_word.jpg', img)
    img = create_words_image(['apple', 'pie'], heats=[0.9, 0.1])
    cv2.imwrite('ig_data/test_word_heatmap.jpg', img)
