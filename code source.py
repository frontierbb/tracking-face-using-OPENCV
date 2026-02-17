import warnings
warnings.filterwarnings('ignore')
import os
import time
import cv2
import numpy as np
from collections import deque, Counter
import urllib.request
from gender_counter import GenderCounter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class EnhancedDetectorV2:
    def __init__(self):
        print("="*60)
        print("ENHANCED HUMAN DETECTOR V2.0")
        print("Improved: Gender | Ethnic | Emotion Detection")
        print("="*60)

        # Initialize face detection
        self.initialize_face_detection()

        # Models
        self.age_gender_enabled = False
        self.age_net = None
        self.gender_net = None
        self.model_files = {
            'age_deploy.prototxt': 'https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt',
            'age_net.caffemodel': 'https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel',
            'gender_deploy.prototxt': 'https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt',
            'gender_net.caffemodel': 'https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel'
        }

        # Tracking
        self.tracked_faces = {}
        self.next_id = 1
        self.person_count = 0
        self.total_detected = 0

        # Performance
        self.frame_skip = 4
        self.frame_count = 0

        # Cache
        self.analysis_cache = {}
        self.cache_duration = 15.0
        self.cache_cleanup_interval = 10.0
        self.last_cache_cleanup = time.time()

        # Temporal smoothing
        self.emotion_history = {}
        self.gender_history = {}
        self.age_history = {}
        self.ethnic_history = {}
        self.history_length = 7
        self.history_cleanup_interval = 15.0
        self.last_history_cleanup = time.time()

        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # FPS
        self.fps_history = deque(maxlen=10)

        # Statistics
        self.emotion_stats = Counter()
        self.gender_stats = Counter()
        self.age_stats = Counter()
        self.ethnic_stats = Counter()

        #gender counter
        self.gender_counter = GenderCounter(cleanup_interval=30.0)

        print("✅ Detector initialized!\n")

    def initialize_face_detection(self):
        """Initialize face detection with dlib landmarks"""
        try:
            import dlib
            self.use_dlib = True
            self.detector = dlib.get_frontal_face_detector()

            if os.path.exists("shape_predictor_68_face_landmarks.dat"):
                self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                print("✅ Dlib with 68 landmarks loaded")
            else:
                print("⚠️  shape_predictor_68_face_landmarks.dat not found!")
                self.predictor = None
                self.use_dlib = False
        except ImportError:
            print("⚠️  Dlib not installed")
            self.use_dlib = False
            self.predictor = None

        # Fallback to OpenCV
        if not self.use_dlib or self.predictor is None:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
            print("✅ Using OpenCV Haar Cascade (fallback)")

    def download_models(self):
        """Download age/gender models if not present"""
        print("\n📥 Checking models...")

        if all(os.path.exists(f) for f in self.model_files.keys()):
            print("✅ All models present!")
            return True

        print("📦 Downloading models (~45MB)...\n")

        for filename, url in self.model_files.items():
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                if size_mb > 0.01:
                    print(f"✅ {filename} ({size_mb:.1f} MB)")
                    continue
                else:
                    os.remove(filename)

            try:
                print(f"⏳ Downloading {filename}...")

                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context

                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

                with urllib.request.urlopen(req) as response:
                    with open(filename, 'wb') as f:
                        f.write(response.read())

                size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"✅ {filename} ({size_mb:.2f} MB)")

            except Exception as e:
                print(f"❌ Failed: {e}")
                return False

        print("\n✅ Download complete!\n")
        return True

    def load_age_gender_models(self):
        """Load age and gender prediction models"""
        if self.age_gender_enabled:
            print("⚠️  Models already loaded!")
            return

        print("\n📥 Loading age/gender models...")

        if not self.download_models():
            print("❌ Model download failed!")
            return

        try:
            print("⏳ Loading age model...")
            self.age_net = cv2.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')

            print("⏳ Loading gender model...")
            self.gender_net = cv2.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt')

            self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            self.gender_list = ['Male', 'Female']

            self.age_gender_enabled = True
            print("✅ Age/Gender models loaded!\n")

        except Exception as e:
            print(f"❌ Loading error: {e}\n")
            self.age_gender_enabled = False

    # ========================================================================
    # IMPROVED FACE DETECTION - Multi-angle support
    # ========================================================================  
    def detect_faces_multi_angle(self, frame, gray):
        """
        Detect faces from multiple angles (frontal + profile)
        Handles edge cases like sideways faces
        """
        all_faces = []
        
        if self.use_dlib and self.predictor:
            # Use dlib detector
            faces = self.detector(gray, 1)  # upsample=1 for better detection
            
            for face in faces:
                x, y = face.left(), face.top()
                w, h = face.right() - x, face.bottom() - y
                
                # Validate coordinates
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                    
                landmarks = self.predictor(gray, face)
                all_faces.append({
                    'bbox': (x, y, w, h),
                    'landmarks': landmarks,
                    'angle': 'frontal'
                })
        else:
            # Multi-cascade approach for better coverage
            
            # 1. Frontal face detection (main)
            faces_frontal = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces_frontal:
                all_faces.append({
                    'bbox': (x, y, w, h),
                    'landmarks': None,
                    'angle': 'frontal'
                })
            
            # 2. Alternative frontal detector (catches difficult angles)
            faces_alt = self.face_cascade_alt.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces_alt:
                # Check if not already detected
                is_duplicate = False
                for existing in all_faces:
                    ex, ey, ew, eh = existing['bbox']
                    # Calculate overlap
                    overlap_x = max(0, min(x+w, ex+ew) - max(x, ex))
                    overlap_y = max(0, min(y+h, ey+eh) - max(y, ey))
                    overlap_area = overlap_x * overlap_y
                    
                    if overlap_area > 0.5 * min(w*h, ew*eh):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_faces.append({
                        'bbox': (x, y, w, h),
                        'landmarks': None,
                        'angle': 'frontal_alt'
                    })
            
            # 3. Profile face detection (left profile)
            faces_profile_left = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces_profile_left:
                # Check if not already detected
                is_duplicate = False
                for existing in all_faces:
                    ex, ey, ew, eh = existing['bbox']
                    overlap_x = max(0, min(x+w, ex+ew) - max(x, ex))
                    overlap_y = max(0, min(y+h, ey+eh) - max(y, ey))
                    overlap_area = overlap_x * overlap_y
                    
                    if overlap_area > 0.3 * min(w*h, ew*eh):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_faces.append({
                        'bbox': (x, y, w, h),
                        'landmarks': None,
                        'angle': 'profile_left'
                    })
            
            # 4. Profile face detection (right profile) - flip image
            gray_flipped = cv2.flip(gray, 1)
            faces_profile_right = self.profile_cascade.detectMultiScale(
                gray_flipped, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces_profile_right:
                # Convert coordinates back to original orientation
                x_original = gray.shape[1] - (x + w)
                
                # Check if not already detected
                is_duplicate = False
                for existing in all_faces:
                    ex, ey, ew, eh = existing['bbox']
                    overlap_x = max(0, min(x_original+w, ex+ew) - max(x_original, ex))
                    overlap_y = max(0, min(y+h, ey+eh) - max(y, ey))
                    overlap_area = overlap_x * overlap_y
                    
                    if overlap_area > 0.3 * min(w*h, ew*eh):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_faces.append({
                        'bbox': (x_original, y, w, h),
                        'landmarks': None,
                        'angle': 'profile_right'
                    })
        
        return all_faces
    
    # ========================================================================
    # ENHANCED HAIR COLOR ANALYSIS (for ethnic detection)
    # ========================================================================
    
    def analyze_hair_color_advanced(self, face_img):
        """
        Advanced hair color analysis
        Returns: dict with hair properties
        """
        h, w = face_img.shape[:2]
        
        # Extract hair region (top 25%)
        hair_region = face_img[:int(h*0.25), :]
        
        if hair_region.size == 0:
            return {
                'color': 'Unknown',
                'darkness': 0,
                'texture': 'Unknown',
                'confidence': 0.0
            }
        
        # Convert to different color spaces
        hair_hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        hair_rgb = cv2.cvtColor(hair_region, cv2.COLOR_BGR2RGB)
        hair_gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        
        # Color properties
        avg_hue = np.mean(hair_hsv[:, :, 0])
        avg_sat = np.mean(hair_hsv[:, :, 1])
        avg_val = np.mean(hair_hsv[:, :, 2])
        avg_r = np.mean(hair_rgb[:, :, 0])
        avg_g = np.mean(hair_rgb[:, :, 1])
        avg_b = np.mean(hair_rgb[:, :, 2])
        
        # Texture analysis
        hair_texture = np.std(hair_gray)
        edges = cv2.Canny(hair_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Classify hair color
        hair_color = 'Unknown'
        darkness_score = 0
        
        # BLACK HAIR (very dark, low saturation)
        if avg_val < 75 and avg_sat < 70:
            hair_color = 'Black'
            darkness_score = 5
        
        # DARK BROWN HAIR
        elif 75 <= avg_val < 115 and avg_sat < 90:
            hair_color = 'Dark Brown'
            darkness_score = 4
        
        # BROWN HAIR
        elif 115 <= avg_val < 145 and 20 < avg_sat < 120:
            if avg_hue < 25:  # Reddish brown
                hair_color = 'Auburn/Reddish Brown'
                darkness_score = 3
            else:
                hair_color = 'Brown'
                darkness_score = 3
        
        # BLONDE HAIR (high value, low saturation)
        elif avg_val >= 145 and avg_sat < 70:
            if avg_val > 180:
                hair_color = 'Platinum Blonde'
                darkness_score = 1
            else:
                hair_color = 'Blonde'
                darkness_score = 2
        
        # RED/GINGER HAIR
        elif avg_hue < 20 and avg_sat > 50 and 100 < avg_val < 180:
            hair_color = 'Red/Ginger'
            darkness_score = 3
        
        # GRAY/WHITE HAIR
        elif avg_sat < 25 and avg_val > 130:
            hair_color = 'Gray/White'
            darkness_score = 1
        
        # Texture classification
        texture_type = 'Unknown'
        if edge_density > 0.15:
            texture_type = 'Curly/Textured'
        elif edge_density > 0.08:
            texture_type = 'Wavy'
        else:
            texture_type = 'Straight'
        
        # Confidence based on hair region quality
        confidence = min(1.0, (hair_region.size / (h * w * 0.25)) * (1.0 if avg_sat > 10 else 0.5))
        
        return {
            'color': hair_color,
            'darkness': darkness_score,
            'texture': texture_type,
            'confidence': confidence,
            'hue': avg_hue,
            'saturation': avg_sat,
            'value': avg_val
        }
        

    # ========================================================================
    # IMPROVED GENDER DETECTION - Multiple Features
    # ========================================================================

    def detect_gender_advanced(self, face_img, landmarks=None):

        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            male_score = 0
            female_score = 0

            # === FEATURE 1: Face Aspect Ratio ===
            face_ratio = w / h if h > 0 else 1.0
            if face_ratio > 0.75:  # Wider face = more male
                male_score += 2.5
            elif face_ratio < 0.65:  # Narrower face = more female
                female_score += 2.5
            else:
                male_score += 1.2
                female_score += 1.2

            # === FEATURE 2: Jawline Analysis ===
            if landmarks is not None:
                # Jaw points: 0-16 (jawline contour)
                jaw_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)]

                # Calculate jaw angle (sharper = male, rounder = female)
                left_jaw = jaw_points[3]
                jaw_bottom = jaw_points[8]
                right_jaw = jaw_points[13]

                # Angle at jaw bottom
                vec1 = np.array([left_jaw[0] - jaw_bottom[0], left_jaw[1] - jaw_bottom[1]])
                vec2 = np.array([right_jaw[0] - jaw_bottom[0], right_jaw[1] - jaw_bottom[1]])

                angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6))
                angle_degrees = np.degrees(angle)

                if angle_degrees < 120:  # Sharp jaw
                    male_score += 3.0
                elif angle_degrees > 140:  # Rounded jaw
                    female_score += 3.0

                # Jaw width variation (male = more square)
                jaw_widths = []
                for i in range(0, 16, 4):
                    left = jaw_points[i]
                    right = jaw_points[16 - i]
                    width = np.linalg.norm(np.array(left) - np.array(right))
                    jaw_widths.append(width)

                jaw_variation = np.std(jaw_widths) / (np.mean(jaw_widths) + 1e-6)
                if jaw_variation < 0.15:  # More uniform = square = male
                    male_score += 2.0
                else:  # More variation = tapered = female
                    female_score += 2.0
            else:
                # Fallback: Use lower face region edge detection
                lower_face = gray[int(h*0.65):, :]
                jaw_sharpness = np.std(lower_face) if lower_face.size > 0 else 30

                if jaw_sharpness > 38:
                    male_score += 2.0
                else:
                    female_score += 1.5

            # === FEATURE 3: Eyebrow Analysis ===
            if landmarks is not None:
                # Eyebrow points: 17-21 (left), 22-26 (right)
                left_brow = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)]
                right_brow = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)]

                # Eyebrow thickness (approximate from y-variation)
                left_thickness = max([p[1] for p in left_brow]) - min([p[1] for p in left_brow])
                right_thickness = max([p[1] for p in right_brow]) - min([p[1] for p in right_brow])
                avg_thickness = (left_thickness + right_thickness) / 2

                # Relative to face height
                thickness_ratio = avg_thickness / h

                if thickness_ratio > 0.04:  # Thicker brows = male
                    male_score += 2.5
                elif thickness_ratio < 0.025:  # Thinner brows = female
                    female_score += 2.5

                # Eyebrow arch (female have more arch)
                left_arch = left_brow[2][1] - (left_brow[0][1] + left_brow[4][1]) / 2
                right_arch = right_brow[2][1] - (right_brow[0][1] + right_brow[4][1]) / 2
                avg_arch = abs((left_arch + right_arch) / 2)

                if avg_arch > 8:  # More arched = female
                    female_score += 2.0
                else:  # Straighter = male
                    male_score += 1.5
            else:
                # Fallback: analyze upper face region
                upper_face = gray[:int(h*0.35), :]
                eyebrow_darkness = np.mean(upper_face) if upper_face.size > 0 else 100

                if eyebrow_darkness < 95:  # Darker/thicker
                    male_score += 1.5
                else:
                    female_score += 1.0

            # === FEATURE 4: Nose Analysis ===
            if landmarks is not None:
                # Nose points: 27-35
                nose_bridge_top = landmarks.part(27)
                nose_tip = landmarks.part(30)
                nose_left = landmarks.part(31)
                nose_right = landmarks.part(35)

                # Nose width
                nose_width = abs(nose_right.x - nose_left.x)
                nose_width_ratio = nose_width / w

                if nose_width_ratio > 0.23:  # Wider nose = male
                    male_score += 2.0
                elif nose_width_ratio < 0.18:  # Narrower = female
                    female_score += 2.0

                # Nose length
                nose_length = abs(nose_tip.y - nose_bridge_top.y)
                nose_length_ratio = nose_length / h

                if nose_length_ratio > 0.15:  # Longer = male
                    male_score += 1.5
                elif nose_length_ratio < 0.12:  # Shorter = female
                    female_score += 1.5

            # === FEATURE 5: Lip Thickness ===
            if landmarks is not None:
                # Lip points: 48-67
                upper_lip_top = (landmarks.part(51).y + landmarks.part(52).y + landmarks.part(53).y) / 3
                upper_lip_bottom = (landmarks.part(61).y + landmarks.part(62).y + landmarks.part(63).y) / 3
                lower_lip_top = (landmarks.part(65).y + landmarks.part(66).y + landmarks.part(67).y) / 3
                lower_lip_bottom = (landmarks.part(57).y + landmarks.part(58).y + landmarks.part(59).y) / 3

                upper_lip_thickness = abs(upper_lip_bottom - upper_lip_top)
                lower_lip_thickness = abs(lower_lip_bottom - lower_lip_top)
                total_lip_thickness = upper_lip_thickness + lower_lip_thickness

                lip_thickness_ratio = total_lip_thickness / h

                if lip_thickness_ratio > 0.08:  # Fuller lips = female
                    female_score += 3.0
                elif lip_thickness_ratio < 0.05:  # Thinner = male
                    male_score += 2.0

            # === FEATURE 6: Cheekbone Prominence ===
            if landmarks is not None:
                # Cheek area: around points 1-3 and 13-15
                face_center_y = (landmarks.part(27).y + landmarks.part(8).y) / 2
                left_cheek_point = landmarks.part(2)
                right_cheek_point = landmarks.part(14)

                avg_cheek_y = (left_cheek_point.y + right_cheek_point.y) / 2
                cheek_height = face_center_y - avg_cheek_y
                cheek_prominence = cheek_height / h

                if cheek_prominence > 0.15:  # High cheekbones = female
                    female_score += 1.0
                else:
                    male_score += 1.0

            # === FEATURE 7: Forehead Analysis ===
            if landmarks is not None:
                # Forehead height (from hairline approx to eyebrows)
                forehead_top_y = min([landmarks.part(i).y for i in range(17, 27)])
                chin_y = landmarks.part(8).y
                face_height = chin_y - forehead_top_y

                # Estimate forehead (from top of frame to eyebrows as proxy)
                forehead_height = forehead_top_y - 0  # Approximate
                forehead_ratio = forehead_height / face_height if face_height > 0 else 0.5

                if forehead_ratio > 0.42:  # Higher forehead = male
                    male_score += 1.5
                elif forehead_ratio < 0.35:  # Lower = female
                    female_score += 1.5

            # === FEATURE 8: Skin Texture (Smoothness) ===
            # Detect edges - more edges = rougher skin = facial hair = male
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            if edge_density > 0.12:  # Rougher = male (facial hair)
                male_score += 2.5
            elif edge_density < 0.08:  # Smoother = female
                female_score += 2.5

            # === FEATURE 9: Hair Analysis ===
            upper_region = gray[:int(h*0.25), :]
            if upper_region.size > 0:
                hair_darkness = np.mean(upper_region)
                hair_variance = np.std(upper_region)

                # Darker, more uniform = short male hair
                if hair_darkness < 90 and hair_variance < 25:
                    male_score += 2.0
                # Lighter, more varied = styled female hair
                elif hair_darkness > 110 or hair_variance > 35:
                    female_score += 1.5

            # === DECISION ===
            total_male = male_score
            total_female = female_score

            # Need significant difference to be confident
            if abs(total_male - total_female) < 3:
                # Too close - return most likely with low confidence marker
                return "Male?" if total_male >= total_female else "Female?"

            return "Male" if total_male > total_female else "Female"


        except Exception as e:
            print(f"Gender detection error: {e}")
            return "Unknown"

    # ========================================================================
    # IMPROVED ETHNIC DETECTION - Hair Color + Nose Bridge Analysis
    # ========================================================================

    def detect_ethnic_advanced(self, face_img, landmarks=None, gray=None):
        try:
            h, w = face_img.shape[:2]
        
            ethnic_scores = {
                'Asian': 0,
                'White': 0,
                'Black': 0,
                'Brown': 0,
                'Middle Eastern': 0
            }
        
        # === FEATURE 1: Hair Color Analysis (PRIMARY - FIXED) ===
            hair_data = self.analyze_hair_color_advanced(face_img)
            hair_color = hair_data['color']
            hair_darkness = hair_data['darkness']
            hair_texture = hair_data['texture']
        
        # Use hair color as strong indicator
            if 'Black' in hair_color:
                ethnic_scores['Asian'] += 5.0
                ethnic_scores['Black'] += 3.5
                ethnic_scores['Brown'] += 3.0
            
            # Texture helps differentiate
                if hair_texture == 'Curly/Textured':
                    ethnic_scores['Black'] += 3.0
                    ethnic_scores['Brown'] += 2.0
                elif hair_texture == 'Straight':
                    ethnic_scores['Asian'] += 3.0
        
            elif 'Dark Brown' in hair_color or 'Brown' in hair_color:
                ethnic_scores['Brown'] += 4.0
                ethnic_scores['Middle Eastern'] += 3.5
                ethnic_scores['White'] += 2.5
                ethnic_scores['Asian'] += 2.0
        
            elif 'Blonde' in hair_color or 'Platinum' in hair_color:
                ethnic_scores['White'] += 6.0
        
            elif 'Red' in hair_color or 'Ginger' in hair_color or 'Auburn' in hair_color:
                ethnic_scores['White'] += 5.0
        
            elif 'Gray' in hair_color or 'White' in hair_color:
            # Don't use for ethnicity (age-related)
                pass
        
        # === FEATURE 2: Nose Bridge Analysis ===
            if landmarks is not None:
                nose_bridge_top = landmarks.part(27)
                nose_tip = landmarks.part(30)
                nose_left = landmarks.part(31)
                nose_right = landmarks.part(35)
            
                bridge_width = abs(nose_right.x - nose_left.x)
                bridge_width_ratio = bridge_width / w
            
                if bridge_width_ratio > 0.22:
                    ethnic_scores['Asian'] += 3.0
                    ethnic_scores['Black'] += 2.5
                elif bridge_width_ratio < 0.18:
                    ethnic_scores['White'] += 3.0
                    ethnic_scores['Middle Eastern'] += 2.0
                else:
                    ethnic_scores['Brown'] += 1.5
        
        # === FEATURE 3: Eye Shape ===
            if landmarks is not None:
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            
                left_eye_width = abs(left_eye[3][0] - left_eye[0][0])
                right_eye_width = abs(right_eye[3][0] - right_eye[0][0])
                avg_eye_width = (left_eye_width + right_eye_width) / 2
            
                left_eye_height = max([p[1] for p in left_eye]) - min([p[1] for p in left_eye])
                right_eye_height = max([p[1] for p in right_eye]) - min([p[1] for p in right_eye])
                avg_eye_height = (left_eye_height + right_eye_height) / 2
            
                eye_aspect = avg_eye_width / (avg_eye_height + 1e-6)
            
                if eye_aspect > 3.5:
                    ethnic_scores['Asian'] += 3.0
                elif eye_aspect < 2.8:
                    ethnic_scores['White'] += 1.5
                    ethnic_scores['Black'] += 1.5
        
        # === FEATURE 4: Skin Tone (MINOR) ===
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            center_region = hsv[int(h*0.4):int(h*0.6), int(w*0.3):int(w*0.7)]
        
            if center_region.size > 0:
                skin_val = np.mean(center_region[:, :, 2])
            
                if skin_val < 90:
                    ethnic_scores['Black'] += 2.0
                elif skin_val > 190:
                    ethnic_scores['White'] += 2.0
                elif 90 < skin_val < 130:
                    ethnic_scores['Brown'] += 1.5
        
        # === DECISION ===
            top_ethnic = max(ethnic_scores, key=ethnic_scores.get)
            top_score = ethnic_scores[top_ethnic]
        
            sorted_scores = sorted(ethnic_scores.values(), reverse=True)
        
            if sorted_scores[0] - sorted_scores[1] < 2:
                top_two = sorted(ethnic_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                return f"{top_two[0][0]}/{top_two[1][0]}"
        
            if top_score < 5:
                return "Mixed"
        
            return top_ethnic
        except Exception as e:
            print(f"Ethnic detection error: {e}")
            return "Unknown"
    # ========================================================================
    # IMPROVED EMOTION DETECTION - Facial Proportions & Ratios
    # ========================================================================

    def detect_emotion_with_proportions(self, landmarks, frame_gray):
        """
        Advanced emotion detection using facial proportions and ratios

        Key improvements:
        1. Compare mouth size to nose size (relative scaling)
        2. Eye size relative to face size
        3. Eyebrow position relative to eye position
        4. Mouth width to face width ratio
        5. Symmetry analysis
        6. Multiple feature ratios
        """
        try:
            # Extract all facial features
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            left_eyebrow = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)]
            right_eyebrow = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)]
            nose = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 36)]
            mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            face_outline = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)]

            # === BASELINE MEASUREMENTS ===
            # Face dimensions
            face_left = face_outline[0][0]
            face_right = face_outline[16][0]
            face_width = face_right - face_left

            face_top = min([landmarks.part(i).y for i in range(17, 27)])
            face_bottom = landmarks.part(8).y
            face_height = face_bottom - face_top

            # Nose size (reference measurement)
            nose_width = abs(nose[8][0] - nose[4][0])  # Nostril width
            nose_height = abs(nose[6][1] - nose[0][1])  # Bridge to tip

            # === RATIO 1: Mouth to Face Width ===
            mouth_left = mouth[0][0]
            mouth_right = mouth[6][0]
            mouth_width = mouth_right - mouth_left

            mouth_to_face_ratio = mouth_width / face_width

            # === RATIO 2: Mouth Opening to Nose Height ===
            mouth_top = min([p[1] for p in mouth])
            mouth_bottom = max([p[1] for p in mouth])
            mouth_height = mouth_bottom - mouth_top

            mouth_to_nose_ratio = mouth_height / (nose_height + 1e-6)

            # === RATIO 3: Eye Opening to Face Height ===
            left_eye_top = min([p[1] for p in left_eye])
            left_eye_bottom = max([p[1] for p in left_eye])
            left_eye_height = left_eye_bottom - left_eye_top

            right_eye_top = min([p[1] for p in right_eye])
            right_eye_bottom = max([p[1] for p in right_eye])
            right_eye_height = right_eye_bottom - right_eye_top

            avg_eye_height = (left_eye_height + right_eye_height) / 2
            eye_to_face_ratio = avg_eye_height / face_height

            # === RATIO 4: Eyebrow to Eye Distance (relative to face) ===
            left_brow_bottom = max([p[1] for p in left_eyebrow])
            left_eye_top_point = min([p[1] for p in left_eye])
            left_brow_eye_dist = left_brow_bottom - left_eye_top_point

            right_brow_bottom = max([p[1] for p in right_eyebrow])
            right_eye_top_point = min([p[1] for p in right_eye])
            right_brow_eye_dist = right_brow_bottom - right_eye_top_point

            avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2
            brow_to_face_ratio = avg_brow_eye_dist / face_height

            # === RATIO 5: Mouth Corner Height Difference ===
            left_mouth_corner = mouth[0]
            right_mouth_corner = mouth[6]
            mouth_center_y = (left_mouth_corner[1] + right_mouth_corner[1]) / 2

            mouth_top_center = mouth[3]
            mouth_bottom_center = mouth[9]

            # Smile/frown indicator: corners vs center
            corner_lift = mouth_center_y - mouth_top_center[1]  # Negative = lifted (smile)
            corner_drop = mouth_bottom_center[1] - mouth_center_y  # Positive = dropped (frown)

            # === RATIO 6: Eye Aspect Ratio (EAR) ===
            def calculate_ear(eye_points):
                A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
                B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
                C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
                return (A + B) / (2.0 * C + 1e-6)

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # === RATIO 7: Mouth Aspect Ratio (MAR) ===
            mouth_top_points = [mouth[13], mouth[14], mouth[15]]
            mouth_bottom_points = [mouth[19], mouth[18], mouth[17]]

            vertical_dists = []
            for i in range(3):
                dist = abs(mouth_bottom_points[i][1] - mouth_top_points[i][1])
                vertical_dists.append(dist)

            avg_mouth_vertical = np.mean(vertical_dists)
            mar = avg_mouth_vertical / (mouth_width + 1e-6)

            # === RATIO 8: Teeth Visibility ===
            teeth_visible = self.detect_teeth_visibility(frame_gray, mouth)

            # === RATIO 9: Symmetry Analysis ===
            # Check if left and right sides are symmetric
            left_right_eye_diff = abs(left_eye_height - right_eye_height)
            left_right_brow_diff = abs(left_brow_eye_dist - right_brow_eye_dist)

            symmetry_score = 1.0 - (left_right_eye_diff + left_right_brow_diff) / face_height

            # ===================================================================
            # EMOTION CLASSIFICATION USING RATIOS
            # ===================================================================

            happy_score = 0
            sad_score = 0
            surprised_score = 0
            angry_score = 0
            neutral_score = 0

            # === HAPPY INDICATORS ===
            # 1. Wide mouth relative to face
            if mouth_to_face_ratio > 0.55:
                happy_score += 3
            elif mouth_to_face_ratio > 0.50:
                happy_score += 2

            # 2. Large mouth opening relative to nose
            if mouth_to_nose_ratio > 0.35:
                happy_score += 3
            elif mouth_to_nose_ratio > 0.25:
                happy_score += 2

            # 3. Teeth visible
            if teeth_visible:
                happy_score += 4

            # 4. Mouth corners lifted
            if corner_lift < -3:  # Negative = lifted
                happy_score += 3

            # 5. Eyes slightly squinted (but not closed)
            if 0.20 < avg_ear < 0.28:
                happy_score += 2

            # 6. Eyebrows neutral or slightly raised
            if brow_to_face_ratio < 0.08:  # Raised
                happy_score += 1

            # 7. Symmetric smile
            if symmetry_score > 0.9:
                happy_score += 1

            # === SAD INDICATORS ===
            # 1. Narrow mouth
            if mouth_to_face_ratio < 0.45:
                sad_score += 2

            # 2. Small mouth opening
            if mouth_to_nose_ratio < 0.15:
                sad_score += 3

            # 3. Mouth corners dropped
            if corner_drop > 3:
                sad_score += 4

            # 4. Eyebrows inner corners raised (sad expression)
            inner_brow_raised = (left_eyebrow[0][1] + right_eyebrow[4][1]) / 2
            outer_brow = (left_eyebrow[4][1] + right_eyebrow[0][1]) / 2
            if inner_brow_raised < outer_brow - 5:
                sad_score += 3

            # 5. Eyes slightly closed
            if avg_ear < 0.22:
                sad_score += 1

            # 6. Low brow position
            if brow_to_face_ratio > 0.10:
                sad_score += 2

            # === SURPRISED INDICATORS ===
            # 1. Wide open eyes
            if avg_ear > 0.30:
                surprised_score += 4

            # 2. Raised eyebrows (far from eyes)
            if brow_to_face_ratio < 0.06:
                surprised_score += 4

            # 3. Large mouth opening
            if mouth_to_nose_ratio > 0.40:
                surprised_score += 3

            # 4. Round mouth (similar width and height)
            mouth_roundness = mouth_height / (mouth_width + 1e-6)
            if mouth_roundness > 0.6:
                surprised_score += 2

            # 5. Eyes very open relative to face
            if eye_to_face_ratio > 0.08:
                surprised_score += 2

            # === ANGRY INDICATORS ===
            # 1. Lowered, furrowed eyebrows
            if brow_to_face_ratio > 0.11:
                angry_score += 4

            # 2. Eyebrows angled down toward center
            left_brow_angle = left_eyebrow[4][1] - left_eyebrow[0][1]
            right_brow_angle = right_eyebrow[0][1] - right_eyebrow[4][1]
            if left_brow_angle > 5 and right_brow_angle > 5:
                angry_score += 3

            # 3. Narrowed eyes
            if avg_ear < 0.23:
                angry_score += 2

            # 4. Tight mouth
            if mouth_to_nose_ratio < 0.18 and mar < 0.20:
                angry_score += 3

            # 5. Lips pressed together
            if mouth_height < nose_height * 0.15:
                angry_score += 2

            # === NEUTRAL BASELINE ===
            # All measurements are in "normal" range
            neutral_score = 2  # Baseline

            if 0.45 < mouth_to_face_ratio < 0.52:
                neutral_score += 1
            if 0.23 < avg_ear < 0.27:
                neutral_score += 1
            if 0.08 < brow_to_face_ratio < 0.10:
                neutral_score += 1
            if abs(corner_lift) < 2 and abs(corner_drop) < 2:
                neutral_score += 1

            # ===================================================================
            # FINAL DECISION
            # ===================================================================

            scores = {
                'Happy': happy_score,
                'Sad': sad_score,
                'Surprised': surprised_score,
                'Angry': angry_score,
                'Neutral': neutral_score
            }

            emotion = max(scores, key=scores.get)
            max_score = scores[emotion]

            # Require minimum threshold
            if max_score < 5 and emotion != 'Neutral':
                emotion = 'Neutral'

            # Return emotion with detailed data
            emotion_data = {
                'ear': avg_ear,
                'mar': mar,
                'mouth_to_face': mouth_to_face_ratio,
                'mouth_to_nose': mouth_to_nose_ratio,
                'brow_to_face': brow_to_face_ratio,
                'teeth': teeth_visible,
                'symmetry': symmetry_score,
                'scores': scores
            }

            return emotion, emotion_data

        except Exception as e:
            print(f"Emotion detection error: {e}")
            return 'Neutral', {}

    def detect_teeth_visibility(self, frame_gray, mouth_points):
        """Detect if teeth are visible"""
        try:
            mouth_hull = cv2.convexHull(np.array(mouth_points))
            mask = np.zeros(frame_gray.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_hull, 255)

            mouth_region = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
            mouth_pixels = mouth_region[mouth_region > 0]

            if len(mouth_pixels) == 0:
                return False

            bright_threshold = np.mean(mouth_pixels) + np.std(mouth_pixels)
            bright_pixels = np.sum(mouth_pixels > bright_threshold)
            teeth_ratio = bright_pixels / len(mouth_pixels)

            return teeth_ratio > 0.15
        except:
            return False

    # ========================================================================
    # REST OF THE CODE (Face Detection, Tracking, UI, etc.)
    # ========================================================================

    def detect_faces_dlib(self, frame, gray):
        """Detect faces using dlib"""
        faces_data = []
        faces = self.detector(gray)

        for face in faces:
            x, y = face.left(), face.top()
            w, h = face.right() - x, face.bottom() - y
            landmarks = self.predictor(gray, face) if self.predictor else None
            faces_data.append({'bbox': (x, y, w, h), 'landmarks': landmarks})

        return faces_data

    def detect_faces_haar(self, frame):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30)
        )

        faces_data = []
        for (x, y, w, h) in faces:
            faces_data.append({'bbox': (x, y, w, h), 'landmarks': None})

        return faces_data

    def analyze_face_complete(self, frame, face_data):
        """Complete face analysis with hybrid approach"""
        x, y, w, h = face_data['bbox']
        landmarks = face_data['landmarks']

        # Extract face image
        padding = 20
        y1 = max(0, y - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(frame.shape[1], x + w + padding)

        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === GENDER DETECTION - HYBRID APPROACH ===
        if self.age_gender_enabled and self.gender_net is not None:
            # Use DNN model (higher accuracy)
            try:
                enhanced = self.preprocess_face(face_img)
                blob = cv2.dnn.blobFromImage(
                    enhanced, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )

                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender_dnn = self.gender_list[gender_preds[0].argmax()]
                gender_confidence = gender_preds[0].max()

                # Get heuristic as backup
                gender_heuristic = self.detect_gender_advanced(face_img, landmarks)

                # Combine: If DNN confidence is high, use it. Otherwise combine both.
                if gender_confidence > 0.75:
                    gender = gender_dnn
                else:
                    # Low confidence - check if heuristic agrees
                    if gender_dnn.replace('?', '') == gender_heuristic.replace('?', ''):
                        gender = gender_dnn  # Both agree
                    else:
                        # Disagreement - use DNN but add marker
                        gender = f"{gender_dnn}*"
            except:
                gender = self.detect_gender_advanced(face_img, landmarks)
        else:
            # Use advanced heuristic method
            gender = self.detect_gender_advanced(face_img, landmarks)

        # === AGE DETECTION - Use DNN if available ===
        if self.age_gender_enabled and self.age_net is not None:
            try:
                enhanced = self.preprocess_face(face_img)
                blob = cv2.dnn.blobFromImage(
                    enhanced, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )

                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                age = self.age_list[age_preds[0].argmax()]
            except:
                # Fallback to heuristic age estimation
                age = self.estimate_age_heuristic(face_img, landmarks)
        else:
            # Heuristic age estimation
            age = self.estimate_age_heuristic(face_img, landmarks)

        # === ETHNIC DETECTION - IMPROVED ===
        ethnic = self.detect_ethnic_advanced(face_img, landmarks, gray)

        # === EMOTION DETECTION - IMPROVED ===
        if landmarks and self.predictor:
            emotion, emotion_data = self.detect_emotion_with_proportions(landmarks, gray)
        else:
            emotion = "Neutral"
            emotion_data = {}

        return {
            'age': age,
            'gender': gender,
            'emotion': emotion,
            'ethnic': ethnic,
            'emotion_data': emotion_data
        }

    def preprocess_face(self, face_img):
        """CLAHE preprocessing for better quality"""
        try:
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        except:
            return face_img
    
    def estimate_age_heuristic(self, face_img, landmarks=None):
        """Heuristic age estimation based on facial features"""
        try:
            h, w = face_img.shape[:2]
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Feature 1: Face size (rough indicator)
            face_area = w * h

            # Feature 2: Skin texture (wrinkles)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            # Feature 3: Skin smoothness
            smoothness = 1.0 - edge_density

            # Feature 4: Face brightness (older = darker usually)
            brightness = np.mean(gray)

            # Scoring
            if landmarks is not None:
                # Use facial proportions
                # Children: bigger eyes relative to face
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                eye_height = max([p[1] for p in left_eye]) - min([p[1] for p in left_eye])
                eye_to_face_ratio = eye_height / h

                if eye_to_face_ratio > 0.08:
                    return "(0-2)" if eye_to_face_ratio > 0.12 else "(4-6)"

            # Age estimation based on texture
            if edge_density > 0.15:  # Many wrinkles
                return "(60-100)" if edge_density > 0.20 else "(48-53)"
            elif edge_density > 0.12:
                return "(38-43)"
            elif edge_density > 0.10:
                return "(25-32)"
            elif edge_density > 0.08:
                return "(15-20)"
            else:
                return "(8-12)"
        except:
            return "(25-32)"  # Default

    def track_faces(self, faces_data):
        """Track faces across frames"""
        new_tracked = {}

        for face_data in faces_data:
            x, y, w, h = face_data['bbox']
            center = (x + w//2, y + h//2)
            min_dist = float('inf')
            best_id = None

            for fid, data in self.tracked_faces.items():
                old_center = data['center']
                dist = np.sqrt((center[0] - old_center[0])**2 + (center[1] - old_center[1])**2)

                if dist < min_dist and dist < 150:
                    min_dist = dist
                    best_id = fid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.total_detected += 1

            new_tracked[best_id] = {
                'bbox': face_data['bbox'],
                'landmarks': face_data['landmarks'],
                'center': center,
                'last_seen': time.time()
            }

        self.tracked_faces = new_tracked
        return new_tracked

    def apply_temporal_smoothing(self, face_id, analysis):
        """Smooth results over time"""
        if not analysis:
            return analysis

        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=self.history_length)
            self.gender_history[face_id] = deque(maxlen=self.history_length)
            self.age_history[face_id] = deque(maxlen=self.history_length)
            self.ethnic_history[face_id] = deque(maxlen=self.history_length)

        self.emotion_history[face_id].append(analysis['emotion'])
        self.gender_history[face_id].append(analysis['gender'])
        self.age_history[face_id].append(analysis['age'])
        self.ethnic_history[face_id].append(analysis['ethnic'])

        emotion_votes = Counter(self.emotion_history[face_id])
        gender_votes = Counter(self.gender_history[face_id])
        age_votes = Counter(self.age_history[face_id])
        ethnic_votes = Counter(self.ethnic_history[face_id])

        return {
            'age': age_votes.most_common(1)[0][0],
            'gender': gender_votes.most_common(1)[0][0],
            'emotion': emotion_votes.most_common(1)[0][0],
            'ethnic': ethnic_votes.most_common(1)[0][0],
            'emotion_data': analysis.get('emotion_data', {})
        }

    def draw_landmarks(self, frame, landmarks):
        """Draw 68 facial landmarks"""
        if not landmarks:
            return

        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            cv2.putText(frame, str(i + 1), (x - 2, y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)

        # Highlight features
        for i in range(36, 48):  # Eyes
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        for i in range(48, 68):  # Mouth
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        for i in range(17, 27):  # Eyebrows
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        for i in range(27, 36):  # Nose
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        #jaw outline
        jaw_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)]
        for i in range(len(jaw_points) - 1):
            cv2.line(frame, jaw_points[i], jaw_points[i + 1], (200, 200, 200), 1)
    def cleanup_histories(self):
        current_time = time.time()
    
    # Get list of currently tracked IDs
        tracked_ids = set(self.tracked_faces.keys())
    
    # Cleanup emotion history
        emotion_ids_to_remove = [fid for fid in self.emotion_history.keys() 
                                if fid not in tracked_ids]
        for fid in emotion_ids_to_remove:
            del self.emotion_history[fid]
    
    # Cleanup gender history
        gender_ids_to_remove = [fid for fid in self.gender_history.keys() 
                                if fid not in tracked_ids]
        for fid in gender_ids_to_remove:
            del self.gender_history[fid]
    
    # Cleanup age history
        age_ids_to_remove = [fid for fid in self.age_history.keys() 
                            if fid not in tracked_ids]
        for fid in age_ids_to_remove:
            del self.age_history[fid]
    
    # Cleanup ethnic history
        ethnic_ids_to_remove = [fid for fid in self.ethnic_history.keys() 
                            if fid not in tracked_ids]
        for fid in ethnic_ids_to_remove:
            del self.ethnic_history[fid]
    
        self.last_history_cleanup = current_time
    
        total_removed = (len(emotion_ids_to_remove) + len(gender_ids_to_remove) + 
                        len(age_ids_to_remove) + len(ethnic_ids_to_remove))
    
        return total_removed        

    # ========================================================================
    # FIXED DRAW DETECTIONS
    # ========================================================================

    def draw_detections(self, frame):
        """Draw all detections"""
        color_primary = (255, 200, 0)
        color_text = (255, 255, 255)
        color_label = (200, 200, 200)

        fram_h, frame_w = frame.shape[:2]

        for face_id, data in self.tracked_faces.items():
            x, y, w, h = data['bbox']
            landmarks = data['landmarks']
            angle = data.get('angle', 'frontal')
        # FIXED: VALIDATE COORDINATES BEFORE DRAWING
            if x < 0 or y < 0 or x + w > frame_w or y + h > fram_h:
                # Clip to frame boundaries
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, fram_h - 1))
                w = max(0, min(w, frame_w - x))
                h = max(0, min(h, fram_h - y))

            # Draw bounding box corners
            length = int(w * 0.2)
            cv2.line(frame, (x, y), (x + length, y), color_primary, 2)
            cv2.line(frame, (x, y), (x, y + length), color_primary, 2)

            cv2.line(frame, (x + w, y), (x + w - length, y), color_primary, 2)
            cv2.line(frame, (x + w, y), (x + w, y + length), color_primary, 2)

            cv2.line(frame, (x, y + h), (x + length, y + h), color_primary, 2)
            cv2.line(frame, (x, y + h), (x, y + h - length), color_primary, 2)

            cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color_primary, 2)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color_primary, 2)
            #DRAW ANGLE
            angle_color = (0, 255, 0) if 'frontal' in angle else (255,165,0)
            cv2.putText(frame, angle[:8], (x, y-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, angle_color, 1, cv2.LINE_AA)

            # Draw landmarks
            if landmarks and self.predictor:
                self.draw_landmarks(frame, landmarks)

            # Get analysis
            cache_key = face_id
            current_time = time.time()

            if cache_key in self.analysis_cache:
                analysis, timestamp = self.analysis_cache[cache_key]
                if current_time - timestamp > self.cache_duration:
                    analysis = self.analyze_face_complete(frame, data)
                    if analysis:
                        if hasattr(self, 'gender_counter'):
                            self.gender_counter.update_gender(face_id, analysis['gender'])
                        analysis = self.apply_temporal_smoothing(face_id, analysis)
                        self.analysis_cache[cache_key] = (analysis, current_time)
            else:
                analysis = self.analyze_face_complete(frame, data)
                if analysis:
                    analysis = self.apply_temporal_smoothing(face_id, analysis)
                    self.analysis_cache[cache_key] = (analysis, current_time)

            if not analysis:
                continue
            self.emotion_stats[analysis['emotion']] += 1
            self.gender_stats[analysis['gender']] += 1
            self.age_stats[analysis['age']] += 1
            self.ethnic_stats[analysis['ethnic']] += 1

            # Info panel
            info_lines = [
                ("AGE", analysis['age']),
                ("GENDER", analysis['gender']),
                ("EMOTION", analysis['emotion']),
                ("ETHNIC", analysis['ethnic'])
            ]

            start_x = x + w + 10
            start_y = y

            # Background
            bg_w, bg_h = 180, 95
            overlay = frame.copy()
            if start_x + bg_w > frame_w:
                start_x = x - bg_w - 10
            if start_y + bg_h > fram_h:
                start_y = fram_h - bg_h - 10

            cv2.rectangle(overlay, (start_x, start_y), (start_x + bg_w, start_y + bg_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # ID
            cv2.putText(frame, f"ID: {face_id:02d}", (start_x + 5, start_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_primary, 1, cv2.LINE_AA)
            cv2.line(frame, (start_x + 5, start_y + 20), (start_x + 40, start_y + 20), color_primary, 1)

            # Info
            for i, (label, value) in enumerate(info_lines):
                line_y = start_y + 38 + (i * 14)
                cv2.putText(frame, f"{label}:", (start_x + 5, line_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_label, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{str(value).upper()}", (start_x + 70, line_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_text, 1, cv2.LINE_AA)

        return frame

    def draw_ui(self, frame, fps, active_ids):
        """Draw UI overlay"""
        height, width = frame.shape[:2]
        
        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Get gender counts
        if hasattr(self, 'gender_counter'):
            current_m, current_f = self.gender_counter.get_current_counts(active_ids=active_ids)
            total_m, total_f = self.gender_counter.get_total_counts()
            
            cv2.putText(frame, 
                       f"Now: {current_m}M / {current_f}F ({current_m + current_f} total)", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(frame, 
                       f"Total: {total_m}M / {total_f}F ({total_m + total_f} unique)", 
                       (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        else:
            cv2.putText(frame, f"Now: {self.person_count} | Total: {self.total_detected}", 
                       (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        
        # FPS
        fps_color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Mode indicator
        mode = "68 Landmarks + Multi-Angle" if (self.use_dlib and self.predictor) else "Multi-Cascade Mode"
        cv2.putText(frame, mode, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if (self.use_dlib and self.predictor) else (255, 200, 0), 1)
        
        # Bottom controls
        cv2.putText(frame, "q:Quit | s:Save | a:LoadDNN | r:Reset | t:Stats | h:Help",
                   (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    def reset_all(self):
        self.tracked_faces = {}
        self.next_id = 1
        self.total_detected = 0
    
    # Cache - FIXED: Now properly cleared
        self.analysis_cache = {}
    
    # Histories - FIXED: Now ALL properly cleared
        self.emotion_history = {}
        self.gender_history = {}
        self.age_history = {}
        self.ethnic_history = {}
    
    # Statistics
        self.emotion_stats = Counter()
        self.gender_stats = Counter()
        self.age_stats = Counter()
        self.ethnic_stats = Counter()
    
    # Gender counter
        if self.gender_counter:
            self.gender_counter.reset()
    
        print("🔄 Reset complete - all data cleared!")

    def show_statistics(self):
        """Show statistics"""
        print("\n" + "="*60)
        print("📊 STATISTICS")
        print("="*60)
        if hasattr(self, 'gender_counter'):
            stats = self.gender_counter.get_stats_dict()
            print(f"\n👥 People:")
            print(f"   Current: {stats['current_male']}M / {stats['current_female']}F ({stats['current_male'] + stats['current_female']} total)")
            print(f"   Lifetime: {stats['total_male']}M / {stats['total_female']}F ({stats['total_unique']} unique)")
            if stats['uncertain'] > 0:
                print(f"   Uncertain: {stats['uncertain']}")
        else:
            print(f"\n👥 People: Current={self.person_count}, Total={self.total_detected}")

        print(f"\n😊 Emotions:")
        for emotion, count in self.emotion_stats.most_common():
            print(f"   {emotion}: {count}")

        print(f"\n👤 Gender:")
        for gender, count in self.gender_stats.most_common():
            print(f"   {gender}: {count}")

        print(f"\n🎂 Age:")
        for age, count in self.age_stats.most_common():
            print(f"   {age}: {count}")

        print(f"\n🌍 Ethnic:")
        for ethnic, count in self.ethnic_stats.most_common():
            print(f"   {ethnic}: {count}")

        print("="*60 + "\n")
    
    def cleanup_analysis_cache(self):
        """
        Dọn dẹp các kết quả phân tích cũ để tránh đầy bộ nhớ (Memory Leak)
        """
        current_time = time.time()
        # Danh sách các ID cần xóa (dữ liệu đã cũ quá 5 giây)
        expired_keys = []
        
        # Duyệt qua cache để tìm dữ liệu cũ
        # self.analysis_cache cấu trúc: {face_id: (data, timestamp)}
        if hasattr(self, 'analysis_cache'):
            for face_id, (data, timestamp) in self.analysis_cache.items():
                # Nếu dữ liệu đã tồn tại hơn 5 giây mà không được cập nhật -> Xóa
                if current_time - timestamp > 5.0:
                    expired_keys.append(face_id)
            
            # Thực hiện xóa
            for key in expired_keys:
                del self.analysis_cache[key]
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 227)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 227)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ Cannot access camera!")
            return

        print("🎥 Camera started!")
        print("💡 Controls: t=Stats, r=Reset, s=Save, q=Quit\n")

        last_cleanup_time = time.time()
        cleanup_interval = 30  # seconds

        screenshot_count = 0
        fps_start = time.time()
        fps_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = time.time()
                if current_time - last_cleanup_time > cleanup_interval:
                    self.cleanup_analysis_cache()
                    if hasattr(self, 'gender_counter'):
                        removed = self.gender_counter.cleanup_inactive()
                        if removed > 0:
                            print(f"🧹 Cleaned up {removed} inactive IDs")
                    last_cleanup_time = current_time
                frame = cv2.flip(frame, 1)

                self.frame_count += 1

                if self.frame_count % self.frame_skip == 0:
                    if self.use_dlib and self.predictor:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces_data = self.detect_faces_multi_angle(frame, gray)
                    self.person_count = len(faces_data)
                    self.track_faces(faces_data)

                frame = self.draw_detections(frame)
                if hasattr(self, 'gender_counter'):
                    for face_id in self.tracked_faces.keys():
                        if face_id in self.analysis_cache:
                            analysis, _ = self.analysis_cache[face_id]
                            if analysis and 'gender' in analysis:
                                self.gender_counter.update_gender(face_id, analysis['gender'])
                if time.time() - self.last_history_cleanup > self.history_cleanup_interval:
                    removed = self.cleanup_histories()
                    if removed > 0:
                        print(f"🧹 Cleaned {removed} history entries")              

                fps_count += 1
                elapsed = time.time() - fps_start

                if elapsed >= 1.0:
                    fps = fps_count / elapsed
                    self.fps_history.append(fps)
                    fps_count = 0
                    fps_start = time.time()

                avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                frame = self.draw_ui(frame, avg_fps, active_ids=self.tracked_faces.keys())

                cv2.imshow('Enhanced Human Detector V2.0', frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n👋 Quitting...")
                    self.show_statistics()
                    break
                elif key == ord('s'):
                    filename = f'detection_{screenshot_count}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved: {filename}")
                    screenshot_count += 1
                elif key == ord('a'):
                    if not self.age_gender_enabled:
                        self.load_age_gender_models()
                elif key == ord('r'):
                    self.reset_all()
                elif key == ord('t'):
                    self.show_statistics()

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
            self.show_statistics()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Goodbye!")

def main():
    print("\n" + "="*60)
    print("ENHANCED HUMAN DETECTOR V2.0")
    print("Improved Gender, Ethnic, Emotion Detection")
    print("="*60 + "\n")

    detector = EnhancedDetectorV2()
    detector.run()

if __name__ == "__main__":
    main()