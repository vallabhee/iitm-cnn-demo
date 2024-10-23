## Solutions for the Self-Test

1. **Basic Python Functions:**
   Write a Python function that takes a list of numbers and returns the list in reverse order.

   **Solution:**

   ```python
   def reverse_list(numbers):
       return numbers[::-1]

   # Example usage
   numbers = [1, 2, 3, 4, 5]
   print(reverse_list(numbers))  # Output: [5, 4, 3, 2, 1]
   ```

   Write a Python function that takes a list of numbers and returns their sum and average.

   **Solution:**

   ```python
   def calculate_stats(numbers):
       return sum(numbers), sum(numbers)/len(numbers)

   # Example usage
   numbers = [1, 2, 3, 4, 5]
   print(calculate_stats(numbers))  # Output: 15, 3
   ```

2. **File Handling:**
   Write Python code to read a file line by line and print each line.

   **Solution:**

   ```python
   def read_file(file_path):
       with open(file_path, 'r') as file:
           for line in file:
               print(line.strip())  # strip() removes any trailing newlines or spaces

   # Example usage (assuming 'example.txt' is in the same directory)
   read_file('example.txt')
   ```

3. **Object-Oriented Programming:**
   Create a class `Animal` with an attribute `name` and a method `speak()` that prints a message.

   **Solution:**

   ```python
   class Animal:
       def __init__(self, name):
           self.name = name

       def speak(self):
           print(f"{self.name} makes a sound.")

   # Example usage
   dog = Animal("Dog")
   dog.speak()  # Output: Dog makes a sound.
   ```

4. **Using Libraries:**
   Import `numpy` and create a 2x3 matrix of ones.

   **Solution:**

   ```python
   import numpy as np

   matrix = np.ones((2, 3))
   print(matrix)

   # Output:
   # [[1. 1. 1.]
   #  [1. 1. 1.]]
   ```

5. **Data Visualization with Matplotlib:**
   Plot a simple line graph using `matplotlib`. The graph should show the relationship between two lists: `x = [1, 2, 3, 4, 5]` and `y = [1, 4, 9, 16, 25]`. Label the axes and give the plot a title.

   ```python
   import matplotlib.pyplot as plt

   # Data
   x = [1, 2, 3, 4, 5]
   y = [1, 4, 9, 16, 25]

   # Plotting the graph
   plt.plot(x, y)

   # Adding labels and title
   plt.xlabel('X-axis: Numbers')
   plt.ylabel('Y-axis: Squares of Numbers')
   plt.title('Line Plot of Numbers vs. Their Squares')

   # Display the plot
   plt.show()
   ```
