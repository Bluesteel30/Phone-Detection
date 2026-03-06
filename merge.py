import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import base64
import json
from PIL import Image
import io
from inference_sdk import InferenceHTTPClient

# --- CONFIGURATION ---
WORKSPACE = "dantes-workspace-va4ef"
WORKFLOW = "sam3-with-prompts"
API_KEY = "*************"
INPUT_IMAGE = "input.jpg"

def process_and_upload(input_image_path):
    PX_PER_CM = 100
    BUFFER_CM = 0.5 
    INTERNAL_W_CM = 10.0  
    INTERNAL_H_CM = 15.0 

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not load {input_image_path}")
        return None

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)

    if ids is not None and len(ids) >= 4:
        centers = {ids[i][0]: np.mean(corners[i][0], axis=0) for i in range(len(ids))}
        if all(k in centers for k in [0, 1, 2, 3]):
            OFFSET = int(BUFFER_CM * PX_PER_CM)
            INT_W, INT_H = int(INTERNAL_W_CM * PX_PER_CM), int(INTERNAL_H_CM * PX_PER_CM)
            TOTAL_W, TOTAL_H = INT_W + (2 * OFFSET), INT_H + (2 * OFFSET)

            pts_src = np.array([centers[0], centers[1], centers[2], centers[3]], dtype="float32")
            pts_dst = np.array([
                [OFFSET, OFFSET],
                [OFFSET + INT_W, OFFSET],
                [OFFSET + INT_W, OFFSET + INT_H],
                [OFFSET, OFFSET + INT_H]
            ], dtype="float32")

            matrix, _ = cv2.findHomography(pts_src, pts_dst)
            aligned_img = cv2.warpPerspective(img, matrix, (TOTAL_W, TOTAL_H))
            
            color_converted = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(color_converted)
            pil_img = pil_img.resize((1000, 1500), Image.Resampling.LANCZOS)
            
            # Save a local copy for your records
            flattened = "flattened.jpg"
            pil_img.save(flattened, "JPEG", quality=65)
            print(f"Success: Image warped and resized.")
            return pil_img # Returns the actual Image object
    
    print("Alignment failed: Markers not found.")
    return None

def call_API(pil_image):
    client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ODUfPahBt2zPD46pMldR"
)
    # 3. Run your workflow on an image
    result = client.run_workflow(
    workspace_name="dantes-workspace-va4ef",
    workflow_id="sam3-with-prompts",
    images={
        "image": pil_image # Path to your image file
    },
    parameters={
        "prompts": ["cell phone","top of cell phone","cell phone in holder"]
    },
    use_cache=True # Speeds up repeated requests
)   
    return(result)

def extract_pixel_locations(data):
    """
    Parses the SAM output. Handles both a single dictionary 
    or a list containing the dictionary.
    """
    # If data is a list, take the first element
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    
    locations = []
    
    # Now data is a dictionary, so .get() will work
    predictions = data.get('sam', {}).get('predictions', [])
    
    for item in predictions:
        x = item.get('x')
        y = item.get('y')
        
        if x is not None and y is not None:
            locations.append((x, y))
            
    return locations

d = {}
count = 1

# Starting position (Top-Left pocket)
start_x = 30 
start_y = 100 

# 1. Increase X-spacing by 20% (from 155 to ~186)
x_spacing = 165

# 2. Triple the Y-buffer logic
# If the previous effective gap was small, we jump to a larger stride
y_spacing = 290 

# Box dimensions remain 120x150
w, h = 120, 150

for y in range(5): # 5 rows
    for x in range(6): # 6 columns
        # Calculate centers using the expanded spacing
        cx = start_x + (x * x_spacing) + (w // 2)
        cy = start_y + (y * y_spacing) + (h // 2)
        
        d[count] = ((cx, cy), (w, h))
        count += 1

def find_matching_grid_zones(predictions, grid_dict):
    matches = []
    
    for pred in predictions:
        # If extract_pixel_locations returns [(x1, y1), (x2, y2)...]
        # pred[0] is x, pred[1] is y
        px = pred[0]
        py = pred[1]
        
        for zone_name, ((gx, gy), (gw, gh)) in grid_dict.items():
            # Calculate the boundaries of the grid zone
            x_min, x_max = gx - (gw / 2), gx + (gw / 2)
            y_min, y_max = gy - (gh / 2), gy + (gh / 2)
            
            # Check if the point is inside the box
            if x_min <= px <= x_max and y_min <= py <= y_max:
                matches.append(zone_name)
                # break  # Uncomment if one point should only match one zone
                
    return matches






def block_match(block):
    block = block.upper()
    if block == 'A':
        d = {1: 'student1', 2: 'student2', 3: 'student3', 4: 'student4', 
        5: 'student5', 6: 'student6', 7: 'student7', 8: 'student8', 9: 'student9', 10: 'student10', 
        11: 'student11', 12: 'student12', 13: 'student13', 14: 'student14', 15: 'student15', 
        16: 'student16', 17: 'student17', 18: 'student18', 19: 'student19', 20: 'student20', 
        21: 'student21', 22: 'student22', 23: 'student23', 24: 'student24', 25: 'student25', 
        26: 'student26', 27: 'student27', 28: 'student28', 29: 'student29', 30: 'student30'}
        return d 
    if block == 'B':
        #random names
        d = {1: 'John', 2: 'Sam', 3: 'Fred', 4: 'Bob', 
        5: 'Frank', 6: 'Kevin', 7: 'Eric', 8: 'Zack', 9: 'Dan', 10: 'James', 
        11: 'Harry', 12: 'Luke', 13: 'Ben', 14: 'Patrick', 15: 'Julian', 
        16: 'Araon', 17: 'Peter', 18: 'Paul', 19: 'Mark', 20: 'Stephen', 
        21: 'Julia', 22: 'Gabriel', 23: 'Matthew', 24: 'Kelly', 25: 'Victoria', 
        26: 'Caroline', 27: 'Katherine', 28: 'Annie', 29: 'Mike', 30: 'Connor'}

        #... for all classes
        #Actual slots for our class
    if block == "E":
        d = {1: 'empty', 2: 'empty', 3: 'Brady L', 4: 'Ethan B', 
        5: 'empty', 6: 'Robbie F', 7: 'Dante D', 8: 'Ethan W', 9: 'empty', 10: 'Harry H', 
        11: 'empty', 12: 'Vince W', 13: 'Narawut K', 14: 'James F', 15: 'empty', 
        16: 'Max O', 17: 'John F', 18: 'Ryan G', 19: 'Henrik L', 20: 'Owen M', 
        21: 'Gabby H', 22: 'Herbert S', 23: 'Yilan Lu', 24: 'Jayden H', 25: 'empty', 
        26: 'Maurice E', 27: 'Sebastian A', 28: 'Isabella W', 29: 'Gianna G', 30: 'Maggie M'}

    return d



def missing_phones(block):
    l = []
    global d
    present_phones = find_matching_grid_zones(extract_pixel_locations(call_API(process_and_upload(INPUT_IMAGE))),d)
    b = block_match(block)
    for key in d:
        if not(key in present_phones or b[key] == "empty"):
            l.append(b[key])
    print(l)

missing_phones("E")


image = cv2.imread("flattened.jpg")

# 3. Draw the zones
for zone_id, ((cx, cy), (w, h)) in d.items():
    # Calculate top-left and bottom-right for CV2 rectangle
    # x_min = center_x - (width / 2)
    x1 = int(cx - (w / 2))
    y1 = int(cy - (h / 2))
    x2 = int(cx + (w / 2))
    y2 = int(cy + (h / 2))

    # Draw the box (Green, thickness 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    q = block_match("E")
    # Label the box with the Zone ID
    cv2.putText(image, q[zone_id], (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 4. Save and show
cv2.namedWindow("Final Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Final Output", 500, 750) 
cv2.imshow("Final Output", image)
cv2.imwrite("output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




