# grader-required-cell

# GRADED FUNCTION: house_model
def house_model():
    ### START CODE HERE

    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([5.0, 10.0, 15.0, 20.0, 25.0], dtype=float)

    # Define your model (should be a model with 1 dense layer and 1 unit)
    # Note: you can use `tf.keras` instead of `keras`
    model = tf.keras.Sequential([keras.layer.Dense(units=1, input_shape=[1])])

    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer="sgd", loss="mean_squared_error")

    # Train your model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs=1000)

    ### END CODE HERE
    return model