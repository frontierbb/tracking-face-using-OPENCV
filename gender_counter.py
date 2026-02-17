import time
from collections import Counter
import cv2
cap = cv2.VideoCapture(0)  # Example video capture, replace as needed

class GenderCounter:
    def __init__(self, cleanup_interval=30.0):
        # Current state
        self.current_genders = {}  # {face_id: "Male"/"Female"}
        self.last_seen = {}  # {face_id: timestamp}
        
        # Historical data
        self.unique_males = set()  # Set of male IDs ever seen
        self.unique_females = set()  # Set of female IDs ever seen
        self.uncertain_ids = set()  # IDs chÆ°a xÃ¡c Ä‘á»‹nh Ä‘Æ°á»£c
        
        # Cleanup
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        self.inactive_timeout = 5.0  # Seconds before considering inactive
    
    def update_gender(self, face_id, smoothed_gender):
        # Update last seen timestamp
        self.last_seen[face_id] = time.time()
        
        # Clean gender string (remove markers)
        clean_gender = self._clean_gender_string(smoothed_gender)
        
        # Handle uncertain cases
        if clean_gender not in ["Male", "Female"]:
            self.uncertain_ids.add(face_id)
            # update last seen
            return "Unknown"
        
        # Remove from uncertain
        self.uncertain_ids.discard(face_id)
        
        # old_gender
        old_gender = self.current_genders.get(face_id)
        
        # Update current gender
        self.current_genders[face_id] = clean_gender
        
        # Update unique sets
        if clean_gender == "Male":
            self.unique_males.add(face_id)
        elif clean_gender == "Female":
            self.unique_females.add(face_id)
        
        # Auto cleanup if needed
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self.cleanup_inactive()
        
        return clean_gender
    
    def _clean_gender_string(self, gender_str):
        if not isinstance(gender_str, str):
            return "Unknown"
        
        # Remove common markers
        clean = gender_str.replace("*", "").replace("?", "").strip()
        
        # Validate
        if clean in ["Male", "Female"]:
            return clean
        else:
            return "Unknown"
    
    def remove_face(self, face_id):
        if face_id in self.current_genders:
            del self.current_genders[face_id]
        
        if face_id in self.last_seen:
            del self.last_seen[face_id]
    def cleanup_inactive(self):
        current_time = time.time()
        inactive_ids = []
        
        for face_id, last_time in self.last_seen.items():
            if current_time - last_time > self.inactive_timeout:
                inactive_ids.append(face_id)
        
        # Remove inactive IDs
        for face_id in inactive_ids:
            self.remove_face(face_id)
        
        # Update cleanup timestamp
        self.last_cleanup = current_time
        
        return len(inactive_ids)  # Return sá»‘ lÆ°á»£ng Ä‘Ã£ xÃ³a
    
    def get_current_counts(self, active_ids=None):
        if active_ids is None:
        # Nếu không truyền active_ids, nó sẽ dùng logic cũ (lỗi)
            count = Counter(self.current_genders.values())
            return count.get("Male", 0), count.get("Female", 0)
        else:
        # CHỈ ĐẾM NHỮNG ID ĐANG XUẤT HIỆN TRÊN FRAME
            males = len([i for i in active_ids if self.current_genders.get(i) == "Male"])
            females = len([i for i in active_ids if self.current_genders.get(i) == "Female"])
            return males, females
    
    def get_total_counts(self):
        return len(self.unique_males), len(self.unique_females)
    
    def get_current_gender(self, face_id):
        return self.current_genders.get(face_id, "Unknown")
    
    def is_gender_confirmed(self, face_id):
        return face_id in self.current_genders
    
    def get_stats_dict(self):
        current_m, current_f = self.get_current_counts()
        total_m, total_f = self.get_total_counts()
        
        return {
            "current_male": current_m,
            "current_female": current_f,
            "total_male": total_m,
            "total_female": total_f,
            "total_unique": total_m + total_f,
            "uncertain": len(self.uncertain_ids)
        }
    
    def reset(self):
        self.current_genders.clear()
        self.last_seen.clear()
        self.unique_males.clear()
        self.unique_females.clear()
        self.uncertain_ids.clear()
        self.last_cleanup = time.time()
    
    def __repr__(self):
        """String representation for debugging"""
        stats = self.get_stats_dict()
        return (f"GenderCounter("
                f"current={stats['current_male']}M/{stats['current_female']}F, "
                f"total={stats['total_male']}M/{stats['total_female']}F)")


# ========================================================================
# INTEGRATION EXAMPLE
# ========================================================================

def integrate_gender_counter_example():
    class EnhancedDetectorWithCounter:
        def __init__(self):
            # ... existing initialization ...
            
            # ADD THIS: Initialize gender counter
            self.gender_counter = GenderCounter(cleanup_interval=30.0)
        
        def run(self):
            # ... main loop ...
            
            while True:
                ret, frame = cap.read()
                
                # ... face detection ...
                
                # Process each tracked face
                for face_id, data in self.tracked_faces.items():
                    # Analyze face
                    analysis = self.analyze_face_complete(frame, data)
                    
                    # Apply temporal smoothing
                    smoothed = self.apply_temporal_smoothing(face_id, analysis)
                    
                    # âœ… UPDATE GENDER COUNTER
                    self.gender_counter.update_gender(face_id, smoothed['gender'])
                
                # Get statistics
                current_m, current_f = self.gender_counter.get_current_counts()
                total_m, total_f = self.gender_counter.get_total_counts()
                
                # Display on frame
                cv2.putText(frame, 
                           f"Now: {current_m}M / {current_f}F", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.putText(frame, 
                           f"Total: {total_m}M / {total_f}F", 
                           (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Detection', frame)
                
                # Handle reset
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    self.gender_counter.reset()
                    print("Gender counter reset!")


# ========================================================================
# TESTS
# ========================================================================

def test_gender_counter():
    """Unit tests for GenderCounter"""
    
    print("Running GenderCounter tests...\n")
    
    counter = GenderCounter()
    
    # Test 1: Basic counting
    print("Test 1: Basic counting")
    counter.update_gender(1, "Male")
    counter.update_gender(2, "Female")
    counter.update_gender(3, "Male")
    
    current_m, current_f = counter.get_current_counts()
    assert current_m == 2 and current_f == 1, "Current count failed"
    
    total_m, total_f = counter.get_total_counts()
    assert total_m == 2 and total_f == 1, "Total count failed"
    print(f" Current: {current_m}M / {current_f}F")
    print(f" Total: {total_m}M / {total_f}F\n")
    
    # Test 2: Marker removal
    print("Test 2: Marker removal")
    counter.update_gender(4, "Male*")
    counter.update_gender(5, "Female?")
    
    current_m, current_f = counter.get_current_counts()
    assert current_m == 3 and current_f == 2, "Marker removal failed"
    print(f" Handles markers: {current_m}M / {current_f}F\n")
    
    # Test 3: Gender flip
    print("Test 3: Gender flip (ID 1: Male â†’ Female)")
    counter.update_gender(1, "Female")
    
    current_m, current_f = counter.get_current_counts()
    assert current_m == 2 and current_f == 3, "Gender flip failed"
    
    # Total should count both (ID 1 was both male and female)
    total_m, total_f = counter.get_total_counts()
    assert total_m == 3 and total_f == 3, "Total after flip failed"
    print(f" After flip - Current: {current_m}M / {current_f}F")
    print(f" Total: {total_m}M / {total_f}F\n")
    
    # Test 4: Face removal
    print("Test 4: Face removal")
    counter.remove_face(1)
    
    current_m, current_f = counter.get_current_counts()
    assert current_m == 2 and current_f == 2, "Face removal failed"
    
    # Total should stay the same
    total_m, total_f = counter.get_total_counts()
    assert total_m == 3 and total_f == 3, "Total should not change"
    print(f" After removal - Current: {current_m}M / {current_f}F")
    print(f" Total unchanged: {total_m}M / {total_f}F\n")
    
    # Test 5: Uncertain gender
    print("Test 5: Uncertain gender")
    counter.update_gender(6, "Unknown")
    counter.update_gender(7, "Uncertain")
    
    current_m, current_f = counter.get_current_counts()
    # Should not change (uncertain not counted)
    assert current_m == 2 and current_f == 2, "Uncertain handling failed"
    print(f" Uncertain IDs ignored: {current_m}M / {current_f}F\n")
    
    # Test 6: Stats dict
    print("Test 6: Full stats dictionary")
    stats = counter.get_stats_dict()
    print(f" Stats: {stats}\n")
    
    # Test 7: Reset
    print("Test 7: Reset")
    counter.reset()
    current_m, current_f = counter.get_current_counts()
    total_m, total_f = counter.get_total_counts()
    
    assert current_m == 0 and current_f == 0, "Reset current failed"
    assert total_m == 0 and total_f == 0, "Reset total failed"
    print(f" Reset successful: {current_m}M / {current_f}F\n")
    
    print("="*60)
    print("ALL TESTS PASSED! âœ…")
    print("="*60)


if __name__ == "__main__":
    # Run tests
    test_gender_counter()
    
    # Show usage example
    print("\n" + "="*60)
    print("USAGE EXAMPLE")
    print("="*60)

