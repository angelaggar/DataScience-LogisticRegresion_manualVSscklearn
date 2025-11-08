# Logistic Regression Comparison: Manual Implementation vs. sklearn

## Description
This project compares a **manual implementation of logistic regression** with the `sklearn` version.  
It includes:

- Training logistic regression from scratch using gradient descent.
- Regularization and class weighting for imbalanced classes.
- Visualization of the sigmoid curve with selected cases (true positives, true negatives, false positives, etc.).
- Evaluation of accuracy on training and test sets.

## Dataset
The project uses the **Breast Cancer Wisconsin (Diagnostic) dataset** (`breast-cancer.csv`):

- The `diagnosis` column is encoded as `M=1` (Malignant) and `B=0` (Benign).  
- Feature columns are normalized before training.

## Requirements
- Python 3
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn
Usage
Train the models and generate predictions:

bash
python mrl_cancer.py       # Manual logistic regression
python mrl_sck.py          # sklearn logistic regression
Visualize selected cases using python mrl_cancer.py :

Graphs are saved in the MCDO/ folder with filenames like curva500loops.png or curva700loops.png.

You can customize which cases to display by editing the indices list in the scripts.

Expected Results
Training accuracy (manual): ~94%

Test accuracy (manual): ~92%

Sigmoid curves with highlighted cases.

Folder Structure
MCDO/
├── README.md
├── breast-cancer.csv
├── mrl_cancer.py       # Manual logistic regression implementation
├── mrl_sck.py          # sklearn logistic regression
├── printer.py          # Visualization functions
├── curva500loops.png   # Example graph for 500 loops
├── curva700loops.png   # Example graph for 700 loops
Notes
The graphs automatically include the number of training loops in the filename.

Use the indices list in the scripts to select cases for visualization (e.g., false positives, true negatives).

License
This is a personal project for learning and comparing logistic regression models.
