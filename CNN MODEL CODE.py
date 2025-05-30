import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

# Step 1: Define the paths to the image directories
benign_dir = 'C:\\Users\\saxen\\OneDrive\\Desktop\\Project\\benign'
malicious_dir = 'C:\\Users\\saxen\\OneDrive\\Desktop\\Project\\malicious'
base_dir = 'C:\\Users\\saxen\\OneDrive\\Desktop\\Project'

# Create directories for training and testing
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Ensure subdirectories exist
benign_train_dir = os.path.join(train_dir, 'benign')
malicious_train_dir = os.path.join(train_dir, 'malicious')
benign_test_dir = os.path.join(test_dir, 'benign')
malicious_test_dir = os.path.join(test_dir, 'malicious')

os.makedirs(benign_train_dir, exist_ok=True)
os.makedirs(malicious_train_dir, exist_ok=True)
os.makedirs(benign_test_dir, exist_ok=True)
os.makedirs(malicious_test_dir, exist_ok=True)

# Function to split data into train and test sets
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_files = os.listdir(SOURCE)
    np.random.shuffle(all_files)
    train_size = int(len(all_files) * SPLIT_SIZE)
    train_set = all_files[:train_size]
    test_set = all_files[train_size:]
    
    for file_name in train_set:
        src_file = os.path.join(SOURCE, file_name)
        dest_file = os.path.join(TRAINING, file_name)
        shutil.copyfile(src_file, dest_file)
    
    for file_name in test_set:
        src_file = os.path.join(SOURCE, file_name)
        dest_file = os.path.join(TESTING, file_name)
        shutil.copyfile(src_file, dest_file)

# Split data into 80% training and 20% testing
split_size = 0.8
split_data(benign_dir, benign_train_dir, benign_test_dir, split_size)
split_data(malicious_dir, malicious_train_dir, malicious_test_dir, split_size)

# Step 2: Preprocess the Data
batch_size = 32
image_size = (15, 1)  # Original size of the images

# Data augmentation and normalization with additional transformations
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,  # Use 20% of the training data for validation
    rotation_range=30,  # Rotate the image within a range
    width_shift_range=0.2,  # Shift the image horizontally
    height_shift_range=0.2,  # Shift the image vertically
    zoom_range=0.2,  # Zoom in on the image
    horizontal_flip=True,  # Flip the image horizontally
    vertical_flip=True,  # Flip the image vertically
    shear_range=0.2  # Shear transformation
)

# Prepare training and validation data generators
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    color_mode='grayscale',  # Since images are 1D, we use grayscale mode
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Prepare test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Verify the class indices and number of classes
print("Classes found:", train_generator.class_indices)
print("Number of classes:", train_generator.num_classes)
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", validation_generator.samples)
print("Number of test samples:", test_generator.samples)

# Ensure there are only two classes
assert train_generator.num_classes == 2, "There should be exactly two classes (benign and malicious)"

# Step 3: Define a More Complex CNN Model for Small Images
def create_model():
    model = Sequential()

    # Add more convolutional and pooling layers to increase model complexity
    model.add(Conv2D(64, (3, 1), activation='relu', input_shape=(15, 1, 1), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 1), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (2, 1), activation='relu', kernel_regularizer=l2(0.001)))  # Adjusted kernel size to (2, 1)
    model.add(BatchNormalization())
    model.add(Dropout(0.25))  # Removed MaxPooling2D to avoid reducing dimensions to zero

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Compile the model
model = create_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Learning rate reduction and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop]
)

# Step 5: Evaluate the Model
# Evaluating the model's performance on the test set
test_steps = test_generator.samples // test_generator.batch_size
Y_test_pred = model.predict(test_generator, test_steps)
y_test_pred = np.round(Y_test_pred).astype(int).flatten()

# Get the true labels
y_test_true = test_generator.classes

# Calculate confusion matrix
cm = confusion_matrix(y_test_true, y_test_pred)
print("Confusion Matrix:\n", cm)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test_true, y_test_pred)
precision = precision_score(y_test_true, y_test_pred)
recall = recall_score(y_test_true, y_test_pred)
f1 = f1_score(y_test_true, y_test_pred)

# Print accuracy, precision, recall, and F1-score
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Step 6: Perform k-fold cross-validation
def get_data_arrays(directory, image_size):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=image_size,
        color_mode='grayscale',
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )
    num_samples = generator.samples
    data = np.zeros((num_samples, image_size[0], image_size[1], 1))
    labels = np.zeros((num_samples,))
    
    for i in range(num_samples):
        data[i], labels[i] = generator.next()
    
    return data, labels

# Load data into arrays
X, y = get_data_arrays(train_dir, image_size)

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

# Perform cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stop],
        verbose=0  # Set verbose to 0 to avoid too much output
    )
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    fold_accuracies.append(val_accuracy)
    
    # Print validation accuracy for each fold
    print(f"Validation Accuracy for Fold {fold+1}: {val_accuracy:.4f}")

# Calculate the average accuracy from the folds
average_accuracy = np.mean(fold_accuracies)
print(f"\nAverage Cross-Validated Accuracy: {average_accuracy:.4f}")
