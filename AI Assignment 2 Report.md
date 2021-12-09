# AI Assignment 2. Report.

# Anna Startseva BS19-04

## 1. Algorithm description

My genetic algorithm simulates oil painting. It uses pictures of real oil strokes.

1. **Chromosomes**
In my genetic algorithm chromosomes are specimens that consist of genes - pictures of oil strokes. The specimen can be represented as an image with images of oil strokes on it, such a picture will have the same size as the input image. Every oil stroke has 6 parameters:
- kind of oil - integer number from 1 to 12 (maximum number of oil samples) which corresponds to the index of the image of oil stroke from dictionary with oil samples. All oil strokes can be found in the attached folder "oil_samples";
- width and height of the oil stroke picture, integers from MIN_OIL_SIZE to MAX_OIL_SIZE (can be from 1 to size of the reference image);
- angle of rotation of the oil stroke picture, integer from 0 to 360 (maximum angle of rotation);
- X-coordinate and Y-coordinate of the oil stroke picture, integers from 0 to 511 (size of the reference image - 1).
    
    So, genes (oil strokes) are lists with dimensions (1, 6); specimens (chromosomes) are lists of lists with dimensions (1, number of oil strokes in the specimen, 6)
    
1. **Population**
The population consists of specimens; it is a list of lists of lists with dimensions (1, number of specimens in the population, number of oil strokes in the specimen, 6). The initial size of the population is 15 because it speeds up calculations.
2. **Fitness function**
Fitness function computes mean square error (MSE) between the reference image and the image that was derived by drawing all oil strokes of the given specimen on an empty picture. Images are transformed to CV2 format - 3-dimensional lists (RGB) of pixels, then sum of squared pixel by pixel difference between CV2 input image and CV2 given image is computed and divided by number of pixels (width * height). I decided to use MSE as fitness function because it was recommended in the book "Hands-on genetic algorithms with Python: applying genetic algorithms to solve real-world deep learning and artificial intelligence problems" [1].
3. **Selection**
Selection process is implemented as selection tournament. It was also recommended in the book [1].  I randomly choose 2 specimens (but this number can vary) from the population by generating 2 random indexes for retrieving specimens from population list (use np.random.choice). These specimens are participants of the selection tournament, and the goal is to choose the best one of them. I compare values of fitness functions of specimens and select the specimen with the lowest value of the fitness function because it satisfies the goal of the algorithm: to minimize the difference between input and produced images. Selected specimen will go through crossover and mutation, and then it will be added to the next population (or its kids). Sequence of these actions will be repeated (population size - 1) times, thus, algorithm will select (population size - 1) specimens, and they can be duplicated but it is normal. One more specimen is chosen from population before selection tournament. This specimen is the best specimen of the previous population (has lowest fitness function value), and it will be added to the new population without any changes (crossover or mutations), so, smallest value of the fitness function in the next population will not be greater than the smallest value of the fitness function from previous population, in other words,  future populations will not be worse than the previous ones. We consider all selected specimens, list of them has the same dimensions as population list.
4. **Crossover**
Crossover is implemented in easy and straightforward way. ****Two consecutive specimens from selected list (iterating through list with step = 2, and choose two consecutive elements) are chosen as parents for crossover. It has 0.9 probability to occur (such probability was recommended in the book [1]), so, if randomly generated number from 0 to 1 is less than or equal to 0.9 crossover is happening. Two kids are composed of parent's genes: first kid has the first half of genes from the first parent, the second half of genes from the second parent; second kid otherwise. If crossover happened kids are added instead of parents to the list of selected specimens, if not - two parents stay in the list of selected without any changes. If number of elements in the population is odd the last specimen is just not considered in the crossover. 
5. **Mutations**
There are three types of mutations. Each type has different probability to be chosen. Three numbers are generated not randomly but with corresponding probabilities: 
1 - 0.4, 2 - 0.2, 3 - 0.4. Then depending on the generated number specific mutation applied to the current specimen (all specimens going through such process):
- 1 - adding new random gen to the specimen. In this mutation, 6 parameters of new oil stroke are generated and added to the specimen;
- 2 - changing position of random gen in the specimen. In this mutation, random gen is chosen, and coordinates from the parameters are replaced with new randomly generated X and Y.
- 3 - changing the type of the random oil stroke in the specimen. In this mutation, random gen is chosen, and index of the oil stroke from the parameters is replaced with new randomly generated index.
    
    Also, all mutations have the same probability to occur, so, at the start of every mutation function random number from 0 to 1 is generated, and if it less than or equal to 0.5 (probability of mutation) mutation occurs, and specimen from the list of selected is replaced by mutated specimen. If mutation doesn't occur specimen remains the same.
    

After all these steps all specimens from list of selected becomes new population and everything repeats number of generations times.

## 2. Examples

### Example 1. Big balloon

Number of generations: 10 000
Initial population size: 15
Initial specimen size: 2000
Size of oil strokes: [10; 20]
Original Images

<img src="https://github.com/asleepann/IAI2021/tree/main/images-for-report/balloon.jpeg" />

Input image

![Output image](https://github.com/asleepann/IAI2021/tree/main/images-for-report/result.png)

Output image

Blurred images

![Input image](https://github.com/asleepann/IAI2021/tree/main/images-for-report/blur_orig.png)

Input image

![Output image](https://github.com/asleepann/IAI2021/tree/main/images-for-report/blur_result.png)

Output image

### Example 2. Flower

Number of generations: 10 000
Initial population size: 15
Initial specimen size: 500
Size of oil strokes: [30; 50]

Original images

![Input image](https://github.com/asleepann/IAI2021/tree/main/images-for-report/flower.jpeg)

Input image

![Output image](https://github.com/asleepann/IAI2021/tree/main/images-for-report/flower10k.png)

Output image

Blurred images

![Input image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur_original.png)

Input image

![Output image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur_gen_10k.png)

Output image

Number of generations: 5260
Initial population size: 15
Initial specimen size: 500
Size of oil strokes: [30; 50]

With minor improvements over the previous version

Original images

![Input image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/flower%201.jpeg)

Input image

![Output image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/flower5260.png)

Output image

Blurred images

![Input image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur_original%201.png)

Input image

![Output image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur_gen_5260.png)

Output image

### Example 3. Fish

Number of generations: 10 000
Initial population size: 15
Initial specimen size: 1000
Size of oil strokes: [20; 40]

Original images

![Input image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/fish.jpeg)

Input image

![Output image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/result%201.png)

Output image

Blurred images

![Input image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur_fish.png)

Input image

![Output image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur_result%201.png)

Output image

### Example 4. Balloons

Number of generations: 10 000
Initial population size: 15
Initial specimen size: 1000
Size of oil strokes: [30; 50]

Original Images

![Input image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/balloons.jpeg)

Input image

![Output image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/result%202.png)

Output image

Blurred images

![Input image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur.png)

Input image

![Output image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur_b.png)

Output image

### Example 5. Head from contest

Number of generations: 8500
Initial population size: 15
Initial specimen size: 2000
Size of oil strokes: [20; 40]

Original images

![Input image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/head.jpg)

Input image

![Output image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/result8500.png)

Output image

Blurred images

![Input image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur_orig%201.png)

Input image

![Output image](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/blur_res.png)

Output image

## 3. What is art for me

In my opinion, art can be any activity that expresses the interests of the author or the viewer. Art can evoke emotions, not necessarily positive or negative, but also ambiguous and incomprehensible to the person experiencing them. Art can make you think about important or mundane and insignificant things. Everyone finds something different in art: the author expresses himself through his creation, he expresses his state, his thoughts; the viewer catches the thoughts that are close to him or finds completely new meanings in the artwork that are important for him at the moment. Art is a reflection not only of the creator but also of the beholder.
Sometimes it happens that something becomes art only because of the viewer, for example, the work of a genetic algorithm. I consider this art not only because I like the resulting images outwardly, but also because I admire how masterfully the computer can imitate works of artists and nature, how the machine with the help of random numbers and algorithms can reproduce images close to reality. I am inspired by the pictures that were made by my algorithm because they prove how wide the scope of application of programming skills is; they can be useful even in creativity, while it seems that there is no place for computer's rationality. I also participated in creating the output images by writing code and tuning parameters, so, in this case, I am both the viewer and the creator. To summarize, the work of the genetic algorithm is the art, because in output pictures I found the reflection of my thoughts about the future, and they inspired me.

## 4. GIFs

If you would like to see gifs of the process you can open this document in browser via link and then look at this section again:

[https://www.notion.so/AI-Assignment-2-Report-08a1c6119f284d718a46aa88b1a9dd74](https://www.notion.so/AI-Assignment-2-Report-08a1c6119f284d718a46aa88b1a9dd74)

![Example 1. Big balloon (101 frames)](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/balloon.gif)

Example 1. Big balloon (101 frames)

![Example 2. Flower (1001 frames)](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/flower.gif)

Example 2. Flower (1001 frames)

![Example 3. Fish (101 frames)](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/fish.gif)

Example 3. Fish (101 frames)

![Example 4. Balloons (101 frames)](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/balloons.gif)

Example 4. Balloons (101 frames)

![Example 5. Head (86 frames)](AI%20Assignment%202%20Report%20544e0e4ffe63433bbb1946a7724b206e/head.gif)

Example 5. Head (86 frames)

## 5. References

1. E. Wirsansky, *Hands-on genetic algorithms with Python: applying genetic algorithms to solve real-world deep learning and artificial intelligence problems.* Birmingham, UK: Packt Publishing Ltd, 2020.

## 6. Code

```python
import copy
import cv2
from PIL import Image
import numpy as np

# CONSTANTS
# Size of the original and result images
WIDTH = 512
HEIGHT = 512
# Number of distinct oil strokes
MAX_OIL_SAMPLES = 12
# Min and max sizes of oil stroke
MIN_OIL_SIZE = 20
MAX_OIL_SIZE = 40
# Max angle of rotation of oil stroke
MAX_ANGLE = 360
# Number of generations
MAX_NUM_GENERATIONS = 10000
# Initial population size
INIT_POPULATION_SIZE = 15
# Initial size of one specimen (number of gens in it)
INIT_SPECIMEN_SIZE = 1000
# Length of gen
NUM_OF_PARAMS = 6
# Probability of crossover
PROB_CROSSOVER = 0.9
# Probability of mutation
PROB_MUTATION = 0.5
# Number of participants in selection tournament
NUM_SELECTION = 2

# Reference image
# Specify name of the image here
# Example: "/.your_image_name.png"
refIm = Image.open("./source_images/head.jpg")

# Oil strokes
im1_orig = Image.open("./oil_samples/1.png")
im2_orig = Image.open("./oil_samples/2.png")
im3_orig = Image.open("./oil_samples/3.png")
im4_orig = Image.open("./oil_samples/4.png")
im5_orig = Image.open("./oil_samples/5.png")
im6_orig = Image.open("./oil_samples/6.png")
im7_orig = Image.open("./oil_samples/7.png")
im8_orig = Image.open("./oil_samples/8.png")
im9_orig = Image.open("./oil_samples/9.png")
im10_orig = Image.open("./oil_samples/10.PNG")
im11_orig = Image.open("./oil_samples/11.PNG")
im12_orig = Image.open("./oil_samples/12.png")

# Resizing of oil strokes to the median of sizes for faster computations
median = (MAX_OIL_SIZE + MIN_OIL_SIZE) // 2
im1 = im1_orig.resize((median, median))
im2 = im2_orig.resize((median, median))
im3 = im3_orig.resize((median, median))
im4 = im4_orig.resize((median, median))
im5 = im5_orig.resize((median, median))
im6 = im6_orig.resize((median, median))
im7 = im7_orig.resize((median, median))
im8 = im8_orig.resize((median, median))
im9 = im9_orig.resize((median, median))
im10 = im10_orig.resize((median, median))
im11 = im11_orig.resize((median, median))
im12 = im12_orig.resize((median, median))

# Dictionary with oil strokes for convenience
# With dictionary we can generate integer between 0 and 12 and easily get oil stroke with the corresponding number
oil_d = {1: im1, 2: im2, 3: im3, 4: im4, 5: im5, 6: im6, 7: im7, 8: im8, 9: im9, 10: im10, 11: im11, 12: im12}

# Drawing of image corresponding to given specimen
def current_image(specimen):
    # You can specify colour of the background by changing 3rd parameter to RGB representation of your desired colour
    # By default it is white (255, 255, 255)
    im0 = Image.new('RGB', (WIDTH, HEIGHT), (255, 255, 255))
    # Drawing of oil strokes one by one
    for i in range(len(specimen)):
        gen = list(map(int, specimen[i]))
        num_oil = gen[0]
        width = gen[1]
        height = gen[2]
        angle = gen[3]
        x = gen[4]
        y = gen[5]
        pict = oil_d[num_oil].resize((width, height))
        pict1 = pict.rotate(angle)
        mask = pict1.convert('L')
        # Pasting oil stroke on the image several times for lower transparency
        # Also, mask is used for removing black background of the strokes
        im0.paste(pict1, (x, y), mask)
        im0.paste(pict1, (x, y), mask)
        im0.paste(pict1, (x, y), mask)
    return im0

# Convert given Pillow image to CV2 format
# CV2 format will help with computing fitness function
def pil_to_cv2(pillow_image):
    return cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)

# Fitness function
# Calculate mean square error (MSE) of difference between the given image and the reference image
def min_square_error(im0, refImCV2):
    return np.sum((pil_to_cv2(im0).astype("float") - refImCV2.astype("float")) ** 2) / float(WIDTH * HEIGHT)

# Generation of gen (one oil stroke)
# It has 6 parameters: kind of the stroke, width and height, angle of rotation, coordinates
def random_gen():
    temp_num_oil = np.random.randint(1, MAX_OIL_SAMPLES + 1)
    temp_width = np.random.randint(MIN_OIL_SIZE, MAX_OIL_SIZE + 1)
    temp_height = np.random.randint(MIN_OIL_SIZE, MAX_OIL_SIZE + 1)
    temp_angle = np.random.randint(0, MAX_ANGLE + 1)
    temp_x = np.random.randint(0, WIDTH + 1)
    temp_y = np.random.randint(0, HEIGHT + 1)
    return [temp_num_oil, temp_width, temp_height, temp_angle, temp_x, temp_y]

# Mutation: add random gen to the given specimen with probability PROB_MUTATION
def mutation_add_gen(specimen):
    chance = np.random.random()
    if chance <= PROB_MUTATION:
        specimen.append(random_gen())

# Mutation: change reposition of random gen in the given specimen with probability PROB_MUTATION
def mutation_change_reposition(specimen):
    chance = np.random.random()
    if chance <= PROB_MUTATION:
        temp_index_gen = np.random.randint(0, len(specimen))
        # Generate new coordinates for random gen and update the old ones
        temp_x = np.random.randint(0, WIDTH + 1)
        temp_y = np.random.randint(0, HEIGHT + 1)
        specimen[temp_index_gen][4] = temp_x
        specimen[temp_index_gen][5] = temp_y

# Mutation: change kind of the random oil stroke in the given specimen with probability PROB_MUTATION
def mutation_change_oil(specimen):
    chance = np.random.random()
    if chance <= PROB_MUTATION:
        temp_index_gen = np.random.randint(0, len(specimen))
        # Generate number (kind) of the stroke and update specimen
        temp_num_oil = np.random.randint(1, MAX_OIL_SAMPLES + 1)
        specimen[temp_index_gen][0] = temp_num_oil

# Crossover of two given specimens
def crossover(specimen1, specimen2):
    chance = np.random.random()
    if chance <= PROB_CROSSOVER:
        length1 = len(specimen1)
        length2 = len(specimen2)
        half1 = length1 // 2
        half2 = length2 // 2
        # kid1 has first half of genes from parent1 and second half of genes from parent2
        # kid2 has first half of genes from parent2 and second half of genes from parent1
        kid1 = [0] * (half1 + (length2 - half2))
        kid2 = [0] * ((length1 - half1) + half2)
        kid1 = copy.deepcopy(specimen1[:half1] + specimen2[-half2:])
        kid2 = copy.deepcopy(specimen2[:half2] + specimen1[-half1:])
        return kid1, kid2
    else:
        return specimen1, specimen2

# Represent reference image in CV2 format
refImCV2 = pil_to_cv2(refIm)

# Compute fitness function for all specimens in the population
# Return list with all computed fitness functions
def all_mse(population):
    fitness_f = []
    for i in range(len(population)):
        current_mse = min_square_error(current_image(population[i]), refImCV2)
        fitness_f.append(current_mse)
    return fitness_f

# Generate initial population
population = []
for i in range(INIT_POPULATION_SIZE):
    # Generate specimen
    # Fill it with random oil strokes
    temp_specimen = []
    for j in range(INIT_SPECIMEN_SIZE):
        temp_specimen.append(random_gen())
    population.append(temp_specimen)

# Main iteration loop of the genetic algorithm
# Consider MAX_NUM_GENERATIONS generations
for i in range(MAX_NUM_GENERATIONS):
    # Print number of generation
    print("Generation:")
    print(i)
    # SELECTION
    # List of selected for crossover and mutation specimens
    best_selected = []
    fitness_function = all_mse(population)
    # Save best specimen with lowest value of the fitness function
    # for adding it to the population at the end of crossover and mutations
    index_best_specimen = np.argmin(fitness_function)
    best_specimen = copy.deepcopy(population[index_best_specimen])
    # Print fitness function of the best specimen (best fitness function in the previous generation)
    print(fitness_function[index_best_specimen])
    # Every 100 generations save drawing of the best specimen to the specified path
    if i % 100 == 0:
        current_image(population[index_best_specimen]).save("C:/Users/Acer/PycharmProjects/labAI/results/result{}.png".format(i))
    # Selection tournament
    # We need to select len(population) - 1 specimens because one best specimen we already have
    for j in range(len(population) - 1):
        # Randomly select NUM_SELECTION number of indexes of participants from population
        selected = np.random.choice(len(population), NUM_SELECTION, replace=False)
        # Find index of participant with lowest fitness function
        min_index = selected[0]
        min_fitness_f = fitness_function[min_index]
        for k in range(len(selected)):
            if fitness_function[selected[k]] < min_fitness_f:
                min_index = selected[k]
                min_fitness_f = fitness_function[selected[k]]
        # Add found specimen to best_selected (using index)
        best_selected.append(population[min_index])
    # CROSSOVER
    # Do crossover providing two consecutive specimens as parents
    for j in range(0, len(best_selected) - 1, 2):
        best_selected[j], best_selected[j + 1] = crossover(best_selected[j], best_selected[j + 1])
    # MUTATION
    for j in range(len(best_selected)):
        # Choose kind of the mutation for j-th specimen
        # Only one mutation can occur
        # Mutations has different probabilities to occur:
        # add gen: 0.4; change reposition of random oil stroke: 0.2;
        # change kind of random oil stroke: 0.4
        mutation_kind = np.random.choice([1, 2, 3], p=[0.4, 0.2, 0.4])
        if mutation_kind == 1:
            mutation_add_gen(best_selected[j])
        elif mutation_kind == 2:
            mutation_change_reposition(best_selected[j])
        else:
            mutation_change_oil(best_selected[j])
    # Add all selected specimens to the next population
    # Also, add best specimen from the previous population to the next population
    best_selected.append(best_specimen)
    population = copy.deepcopy(best_selected)

# Save result (drawing of the best specimen with lowest value of the fitness function from last population)
# to the specified path
result_index = np.argmin(all_mse(population))
current_image(population[result_index]).save("C:/Users/Acer/PycharmProjects/labAI/results/result.png")
```
