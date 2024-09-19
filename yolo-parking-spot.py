import pandas as pd
import sys
import os

from utils.functions import process_images, get_results_df, mean_df
from utils.functions import plot_model_size, plot_model_params, plot_precision_recall, plot_mAP, save_plots

# Get current working directory
# repo_path = os.path.dirname(os.getcwd())
repo_path = os.getcwd()


# Path to the CSV file
filename = f'{repo_path}/models/models_summary.csv'

# Import the CSV file into a pandas DataFrame
models_summary = pd.read_csv(filename, thousands='.', decimal=',')

# Show the models summary
models_summary.sort_values(by=['mAP50-95'], ascending=False).head(8)


# Specify the folder path containing the images and annotations
data_path = os.path.join(repo_path, 'data')

# Specify the output folder path
output_folder = os.path.join(repo_path, 'results/without_yolo/')

# process_images returns a dataframe containing the results
without_yolo_results = process_images(
    data_path, output_folder, highlighted_cars=True)
without_yolo_results.head(5)  # Show the results for the first 5 images


# Specify the model name
model = 'yolov5n_fold_0'

# Specify the folder path containing the images and annotations
data_path = os.path.join(repo_path, 'data')

# Specify the output folder path
output_folder = f'{repo_path}/results/{model}/'

# process_images returns a dataframe containing the results
yolov5n_results = process_images(
    data_path, output_folder, highlighted_cars=True, model=model)
yolov5n_results.head(5)  # Show the results for the first 5 images


model = 'yolov5s_fold_0'
output_folder = f'{repo_path}/results/{model}/'
yolov5s_results = process_images(
    data_path, output_folder, highlighted_cars=True, model=model)
yolov5s_results.head()  # Show the results for the first 5 images


# Specify the model name
model = 'yolov8n_fold_0'

# Specify the folder path containing the images and annotations
data_path = os.path.join(repo_path, 'data')

# Specify the output folder path
output_folder = f'{repo_path}/results/{model}/'

# process_images returns a dataframe containing the results
yolov8n_results = process_images(
    data_path, output_folder, highlighted_cars=True, model=model)
yolov8n_results.head()  # Show the results for the first 5 images


model = 'yolov8s_fold_0'
output_folder = f'{repo_path}/results/{model}/'
yolov8s_results = process_images(
    data_path, output_folder, highlighted_cars=True, model=model)
yolov8s_results.head()  # Show the results for the first 5 images


# Specify the model path or the model name if it is in the models folder.
# model_path = f'{repo_path}models/yolov5n/runs/exp/weights/best.pt'
# model = "my_yolovXX.pt"

# Specify the output folder path
# output_folder = f'{repo_path}/results/{model}/'

# process_images returns a dataframe containing the results
# results = process_images(data_path, output_folder, highlighted_cars=True, model=model_path)
# results.head(20) # # Show the results for the first 20 images


models_summary = mean_df(models_summary)
models_summary.head()
# Plot the model size
plot_model_size(models_summary)
plot_model_params(models_summary)
plot_precision_recall(models_summary)
plot_mAP(models_summary)
save_plots()
results_df = get_results_df(without_yolo_results, yolov5n_results,
                            yolov5s_results, yolov8n_results, yolov8s_results)
results_df.head()
