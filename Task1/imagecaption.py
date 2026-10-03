import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = 'Salesforce/blip-image-captioning-base'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_processor = None
_model = None


def _load():
    """Lazy-load the BLIP processor and model (first call downloads/caches)."""
    global _processor, _model
    if _model is None:
        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        # use_safetensors=True: this transformers release refuses to load .bin
        # checkpoints because torch.load here predates the fixed torch version.
        _model = BlipForConditionalGeneration.from_pretrained(
            MODEL_NAME, use_safetensors=True
        ).to(DEVICE)
        _model.eval()
    return _processor, _model


def get_caption(image, max_new_tokens=64):
    """Return an English caption for a PIL image or an image file path."""
    processor, model = _load()
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')
    else:
        image = image.convert('RGB')

    inputs = processor(image, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        token_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.batch_decode(token_ids, skip_special_tokens=True)[0]
    return caption.strip()


if __name__ == '__main__':
    import os
    import sys

    if len(sys.argv) != 2:
        print('Usage: python3 imagecaption.py <path_to_image>')
        sys.exit(1)
    if not os.path.isfile(sys.argv[1]):
        print(f'Image file not found: {sys.argv[1]}')
        sys.exit(1)
    print(get_caption(sys.argv[1]))
