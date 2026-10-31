import tensorflow as tf
# Load images
data = tf.keras.utils.image_dataset_from_directory(
    r"vehical classification/train",
    image_size=(64, 64),
    batch_size=2
)
# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation="softmax")
])
# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Train
model.fit(data, epochs=50)
model.summary()
# Prediction
img = tf.keras.utils.load_img("vehical classification/test/P1.png", 
                              target_size=(64,64))
x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, 0)

p = model.predict(x)
print(data.class_names[p.argmax()])
model.save('CAT.keras')

import json

# Save class names to a JSON file
with open("classes.json", "w") as f:
    json.dump(data.class_names, f)
    import json

# Read class names back from the JSON file
with open("classes.json", "r") as f:
    class_names = json.load(f)

import tensorflow as tf
m=tf.keras.models.load_model("cat.keras")
img = tf.keras.utils.load_img("vehical classification/test/P1.png", 
                              target_size=(64,64))
x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, 0)
p=m.predict(x)
print(p)
print(class_names[p.argmax()])