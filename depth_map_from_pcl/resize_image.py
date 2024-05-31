from PIL import Image

def resize_images_to_same_size(image1_path, image2_path):
    """
    Resize two images to the same size, based on the smaller dimensions of the two.
    
    Parameters:
    image1_path (str): Path to the first image.
    image2_path (str): Path to the second image.
    output1_path (str): Path to save the resized first image.
    output2_path (str): Path to save the resized second image.
    """
    # Open the images
    image_1 = Image.open(image1_path)
    image_2 = Image.open(image2_path)
    
    # Get the dimensions of the images
    width_1, height_1 = image_1.size
    width_2, height_2 = image_2.size
    
    dimensions = [width_1, height_1, width_2, height_2]
    minimum = max(dimensions)
    index_max = dimensions.index(minimum)
    
    # Determine the new size based on the smaller dimensions
    new_width = 0
    new_height = 0
    
    if index_max<2:
        # resize first image
        if index_max==0:
            # width
            new_height = height_2
            new_width = int((height_2/height_1)*width_1)
        else:
            # height
            new_width = width_2
            new_height = int((width_2/width_1)*height_1)
        resized_image = image_1.resize((new_width, new_height), Image.LANCZOS)
        
        resized_image.save("resized_image.png")
        image_2.save("image_not_resized.png")
        print(f"Not resized image dimension {width_2}x{height_2}")
        print(f"Image has been resized to {new_width}x{new_height}")
    else:
        # resize second image
        if index_max==2:
            # width
            new_height = height_1
            new_width = int((height_1/height_2)*width_2)
        else:
            # height
            new_width = width_1
            new_height = int((width_1/width_2)*height_2)
       
        resized_image = image_2.resize((new_width, new_height), Image.LANCZOS)
        resized_image.save("resized_image.png")
        image_1.save("image_not_resized.png")
        print(f"Not resized image dimension {width_1}x{height_1}")
        print(f"Image has been resized to {new_width}x{new_height}")

        
    


# Esempio di utilizzo:
resize_images_to_same_size('image1.png', 'image2.png')
