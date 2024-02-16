from PIL import Image, ImageDraw, ImageFont
from utils import *
from torchvision import transforms
import torch
from model.SSD300 import SSD300
from model.vgg import VGG16BaseNet, AuxiliaryNet, PredictionNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_score = 0.4
max_overlap = 0.45
top_k = 200
img_path = "./test/image.jpg"
trained_model = torch.load("model_state_ssd300.pth.tar")
output_path = "./test/"
start_epoch = trained_model["epoch"] + 1
print('\nLoaded model trained with epoch %d.\n' % start_epoch)
model = trained_model['model']
model = model.to(device)
model.eval()

resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect(model, device, original_image, min_score, max_overlap, top_k, suppress = None):

    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)

    locs_pred, cls_pred = model(image.unsqueeze(0))

    detect_boxes, detect_labels, detect_scores = model.detect(locs_pred, cls_pred,
                                                              min_score, max_overlap, top_k)

    detect_boxes = detect_boxes[0].to('cpu')

    original_dims = torch.FloatTensor(
            [original_image.width, original_image.height, original_image.width,
             original_image.height]).unsqueeze(0)

    detect_boxes = detect_boxes * original_dims

    detect_labels = [rev_label_map[l] for l in detect_labels[0].to('cpu').tolist()]

    if detect_labels == ["background"]:
        return original_image

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("arial.ttf", 15)

    for i in range(detect_boxes.size(0)):
        if suppress is not None:
            if detect_labels[i] in suppress:
                continue

        box_location = detect_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[detect_labels[i]])

        draw.rectangle(xy=[l + 1. for l in box_location], outline=
                       label_color_map[detect_labels[i]])

        text_size = font.getsize(detect_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[detect_labels[i]])
        draw.text(xy=text_location, text=detect_labels[i].upper(), fill='white',font=font)

    print("Detect label: ", detect_labels)
    return annotated_image


if __name__ == '__main__':
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    annotated_image = detect(model, device, original_image, min_score=args.min_score,
                             max_overlap=args.max_overlap, top_k= args.top_k)
    annotated_image.save(ouput_path)
    annotated_image.show()
