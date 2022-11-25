import face_recognition as fc
f=fc.load_image_file("C:/Users/Subham/Desktop/Learn N Build/Image processing/face recognition/white.JPG")
my_face_f=fc.face_encodings(f)[0]
test_image=fc.load_image_file("C:/Users/Subham/Desktop/Learn N Build/Image processing/face recognition/WIN_20220905_21_08_05_Pro.jpg")
test_image_f=fc.face_encodings(test_image)[0]
result=fc.compare_faces([my_face_f],test_image_f)
print(result)
