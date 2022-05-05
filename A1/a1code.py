
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math
import numpy as np
from skimage import io

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, n_channels).
    """


    # out = None
    # YOUR CODE HERE
    # my_img = io.imread(path)
    # out = np(my_img)
    out = io.imread(img_path)/255
#     print(out)


    return out

def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none

                
    """
    # YOUR CODE HERE
    # print(image.ndim)
    
    if image.ndim == 3:
        height, width, channel = image.shape
    else:
        height, width = image.shape
        channel = 1
    # height, width, channel = image.shape 

    print("height:", height)
    print("width:", width  )
    print("channel:", channel)
    print()

    
    return None

def crop(image, x1, y1, x2, y2):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:0609
        image: numpy array of shape(image_height, image_width, 3).
        (x1, y1): the coordinator for the top-left point
        (x2, y2): the coordinator for the bottom-right point
        

    Returns:
        out: numpy array of shape(x2 - x1, y2 - y1, 3).
    """

    out = None
    out = image[y1:y2, x1:x2]

    ### YOUR CODE HERE

    return out
    
def resize(input_image, fx, fy):
    """Resize an image using the nearest neighbor method.
    Not allowed to call the matural function.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.
        fx (float): the resize scale on the original width.
        fy (float): the resize scale on the original height.

    Returns:
        np.ndarray: Resized image, with shape `(image_height * fy, image_width * fx, 3)`.
    """
    # out = [[[]]]
    height, width, channel = input_image.shape
    new_y = int(height*fy)
    new_x = int(width*fx)
    # print(height,"*",fy," is",new_y)
    # print(width,"*",fx," is",new_x)
    out = np.zeros((new_y,new_x,3))
    # print(out)
    for y in range(0,new_y):
        for x in range(0,new_x):
            if math.floor(y/fy) >= height:
                the_y = height-1
            else:
                the_y = math.floor(y/fy)
                
            if math.floor(x/fx) >= width:
                the_x = width-1
            else:
                the_x = math.floor(x/fx)
                
            out[y][x] = input_image[the_y][the_x]
                        # out[x][y] = input_image[int(x)][int(y)]
    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following
    

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, divided by 255.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    height, width, channel = image.shape
    out = np.zeros((height,width,channel))
    for x in range(height):
        for y in range(width):
            for z in range(channel):
                out[x][y][z] = factor*(image[x][y][z]-0.5)+0.5
    return out

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(image_height, image_width)`.
    """
    height, width, channel = input_image.shape
    out = np.zeros((height,width,1))
    for x in range(height):
        for y in range(width):
            # for z in range(channel):
            #     out[x][y][z] = factor*(image[x][y][z]-0.5)+0.5
            average = (input_image[x][y][0]+input_image[x][y][1]+input_image[x][y][2])/3
#             out[x][y][0] = out[x][y][2] = out[x][y][1] = average
            out[x][y]= average
            
    return out
    
def binary(grey_img, th):
    """Convert a greyscale image to a binary mask with threshold.
  
                  x_n = 0, if x_p < th
                  x_n = 1, if x_p > th
    
    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        th (float): The threshold used for binarization, and the value range is 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    height, width, channel = grey_img.shape
    out = np.zeros((height,width,channel))
    for x in range(height):
        for y in range(width):
            for z in range(channel):
                if grey_img[x][y][z] < th:
                    out[x][y][z] = 0
                elif grey_img[x][y][z] > th:
                    out[x][y][z] = 1
                else:
                    out[x][y][z] = th
                        
    return out

def my_pad(image,height_board,width_board):
    img_h,img_w = image.shape
    out = np.zeros((img_h+2*height_board,img_w+2*width_board))
    for x in range(img_h):
        for y in range(img_w):
            out[height_board+x][width_board+y] = image[x][y]
            
    return out 


def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    # out = None
    
    ### YOUR CODE HERE
    img_h , img_w = image.shape
    ker_h , ker_w = kernel.shape
    out = np.zeros((img_h,img_w))
    # width_board = int(np.floor((img_w - ( img_w - ker_w + 1)) / 2))
    # height_board = int(np.floor((img_h - ( img_h - ker_h + 1)) / 2))
    width_board = int((ker_w-1)/2)
    height_board = int((ker_h-1)/2)
    
    reverse_kernel = np.flip(kernel)
    # reverse_kernel = np.zeros((ker_h,ker_w))
    # for h_ker in range(ker_h):
    #     for w_ker in range(ker_w):
    #         reverse_kernel[h_ker][w_ker] = kernel[ker_h-1- h_ker][ker_w -1- w_ker] 
#     print(image.shape)
#     print("board:",height_board,width_board)
    
    # img_pad = np.pad(image,(height_board,width_board))
    img_pad = my_pad(image,height_board,width_board)
#     print(img_pad.shape)
    
#     print(img_h+ker_h)
#     print(img_w+ker_w)
    for the_h in range(int(img_h)):
        for the_w in range(int(img_w)):
            num_count = 0
            out[the_h,the_w] = sum([img_pad[the_h+h,the_w+w]*reverse_kernel[h,w] for h in range(ker_h) for w in range(ker_w)] )
    return out

    
    
    
    
#     img_h , img_w = image.shape
# #     print(img_w)
#     ker_h , ker_w = kernel.shape
#     out = np.zeros((img_h,img_w))
#     width_board = int((img_w-(img_w-ker_w+1))/2)
#     height_board = int((img_h-(img_h - ker_h +1))/2)
#     reverse_kernel = np.zeros((ker_h,ker_w))
#     for h_ker in range(ker_h):
#         for w_ker in range(ker_w):
#             reverse_kernel[h_ker][w_ker] = kernel[ker_h-1- h_ker][ker_w -1- w_ker] 
# #     print(reverse_kernel)
#     for the_h in range(int(img_h-ker_h+1)):
#         for the_w in range(int(img_w-ker_w+1)):
#             num_count = 0
#             out[height_board+the_h,width_board+the_w] = sum([image[the_h+h,the_w+w]*reverse_kernel[h,w] for h in range(ker_h) for w in range(ker_w)] )

# #     print(out)
#     return out

def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1
#     print(test_img)

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3
#     display(expected_output)
    # print()
#     print(expected_output)
#     display(test_output)
    # Test if the output matches expected output
#     assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."
    return test_output


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    # out = None
    ### YOUR CODE HERE
    # print(image.ndim)
    # if image.ndim != 3:
    #     out = conv2D(image,kernel)
    #     return out
    # else:
    
    img_h , img_w , img_d = image.shape
    # print(image,"hello")
    if img_d == 1 :
        # d = np.dsplit(image,1)
        # print(d)
        d = image.reshape(img_h,img_w)
        return conv2D(d,kernel)
        
    a,b,c = np.dsplit(image,3)
#     print(a)
#     a_h,a_w,a_d = a.shape

    a = a.reshape(img_h,img_w)
    b = b.reshape(img_h,img_w)
    c = c.reshape(img_h,img_w)

#     print(a)
#     print(b)
#     print(c)
    # print(a.shape,kernel.shape)
    a = conv2D(a,kernel)
    b = conv2D(b,kernel)
    c = conv2D(c,kernel)

#     print(a)
#     print(b)
#     print(c)
#     out = np.dstack((a,b))
#     out = np.dstack((out,c))
    out = np.stack([a,b,c], axis =2)
#     out = np.stack([out,c], axis = 2)
#     print(out)
    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    # out = None
    ### YOUR CODE HERE
    if kernel.ndim == 3:
        ker_h, ker_w, ker_d = kernel.shape
        r_kernel = kernel.reshape(ker_h,ker_w)
        r_kernel = np.flip(r_kernel)    
    else:
        r_kernel = np.flip(kernel)
    out = conv(image,r_kernel)
    out = out/np.amax(out)
    
    
    # print(np.unravel_index(np.argmax(out,axis=None),out.shape))
    
    return out


#5.3



#4.2 code 
def gaussian_pyramid(image):
    img_h, img_w, img_d = image.shape
    # out = np.zeros((img_h,img_w,img_d))
    out = np.zeros((math.floor(img_h/2),math.floor(img_w/2),int(img_d)))
    h = w = 0
    for x in range(1,img_h,2):
        w=0
        for y in range(1,img_w,2):
            out[h][w]= image[x][y]
            w=w+1
        h=h+1
        
                    
    return out

# question 5 area_show
def area_show(image,point,target_size):
    target_row, target_width ,n= target_size.shape
    target_row = (target_row-1)/2
    target_width = (target_width-1)/2
    point_x,point_y = point
    image_row, image_width , n = image.shape
    
    l_bound = point_x - target_row
    r_bound = point_x + target_row
    t_bound = point_y - target_width
    b_bound = point_y + target_width
    
    white_line =3
    
    out= np.zeros((image_row,image_width))
    for x in range(image_row):
        for y in range(image_width):
            # if((abs(x - l_bound) < white_line or abs(x - r_bound) <white_line) and (abs(y - t_bound) <white_line or abs(y - b_bound) < white_line) ):
            #     out[x][y] = 1
            # if ((l_bound-white_line < x < r_bound+white_line ) and (y == t_bound or y == b_bound)):
            #     out[x][y] = 1
            #     continue
            # if((x == l_bound or x == r_bound) and (t_bound-white_line < y < b_bound+white_line)):
            #     out[x][y] = 1
            #     continue
            # out[x][y] = image[x][y]
            if ((l_bound-white_line < x < r_bound+white_line ) and (abs(y-t_bound)<white_line or abs(y-b_bound)<white_line)):
                out[x][y] = 1
                continue
            if((abs(x-l_bound)<white_line or abs(x-r_bound)<white_line) and (t_bound-white_line < y < b_bound+white_line)):
                out[x][y] = 1
                continue
            out[x][y] = image[x][y]
    return out
    
    