import os
from PIL import Image
import torch
from torchvision import transforms
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load pretrained models
tf_model = ResNet50(weights='imagenet')
torch_model = models.resnet50(pretrained=True)
torch_model.eval()

# Define the path to the folder containing images
image_folder = 'pet_images'

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        # For TensorFlow
        tf_img = image.load_img(os.path.join(image_folder, filename), target_size=(224, 224))
        tf_img_array = image.img_to_array(tf_img)
        tf_img_array = preprocess_input(tf_img_array.reshape(1, 224, 224, 3))
        tf_predictions = tf_model.predict(tf_img_array)
        tf_decoded_predictions = decode_predictions(tf_predictions)
        print(f"\n** TensorFlow Predictions for {filename}: **")
        print(tf_decoded_predictions)

        # For PyTorch
        torch_img = Image.open(os.path.join(image_folder, filename))
        torch_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        torch_img_tensor = torch_preprocess(torch_img)
        torch_img_tensor = torch.unsqueeze(torch_img_tensor, 0)
        with torch.no_grad():
            torch_output = torch_model(torch_img_tensor)
        print(f"\n** PyTorch Predictions for {filename}: **")
        probabilities = torch.nn.functional.softmax(torch_output[0], dim=0).numpy()
        top_classes = torch.topk(torch_output, k=5)
        top_classes_indices = top_classes.indices.numpy()

print(f"\n** Top Predicted Classes and Probabilities for {filename}: **")
for i in range(top_classes_indices.shape[1]):
    class_index = top_classes_indices[0, i]
    class_probability = probabilities[class_index]
    print(f"Class {class_index}: Probability {class_probability:.4f}")




    print("\n** Summary Results for {} Model **".format(model.upper()))
    print("Number of Images: {}".format(results_stats_dic['n_images']))
    print("Number of Dog Images: {}".format(results_stats_dic['n_dogs_img']))
    print("Number of Not-a-Dog Images: {}".format(results_stats_dic['n_notdogs_img']))
    
    for stat_name, stat_value in results_stats_dic.items():
        if stat_name.startswith('pct'):
            print("{}: {:.2f}%".format(stat_name, stat_value))
    
        if print_incorrect_dogs and (results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_notdogs'] != results_stats_dic['n_images']):
         print("\n** Incorrect Dog Classifications: **")
        for key, value in results_dic.items():
            if sum(value[3:]) == 2 and value[2] == 0:
                print("Real: {}, Classifier: {}".format(value[0], value[1]))
    
    if print_incorrect_breed and results_stats_dic['n_correct_dogs'] != results_stats_dic['n_correct_breed']:
        print("\n** Incorrect Breed Classifications: **")
        for key, value in results_dic.items():
            if sum(value[3:]) == 2 and value[2] == 0 and value[3] == 1:
                print("Real: {}, Classifier: {}".format(value[0], value[1]))
