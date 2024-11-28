from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import authenticate, login as auth_login
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
import os


def home(request):
    return render(request, 'home.html')


def login(request):
    if request.method == "POST":
        un = request.POST['username']
        pw = request.POST['password']
        user = authenticate(request, username=un, password=pw)
        if user is not None:
            auth_login(request, user)
            return redirect('/profile')  # Redirect to profile after successful login
        else:
            msg = 'Invalid Username/Password'
            form = AuthenticationForm()
            return render(request, 'login.html', {'form': form, 'msg': msg})
    else:
        form = AuthenticationForm()
        return render(request, 'login.html', {'form': form})


def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/login')  # Redirect to the login page after successful signup
    else:
        form = UserCreationForm()

    return render(request, 'signup.html', {'form': form})


def profile(request):
    if request.method == 'POST' and request.FILES.get('uploaded_image'):
        uploaded_image = request.FILES['uploaded_image']

        # Validate the uploaded file type
        if not uploaded_image.content_type.startswith('image/'):
            return render(request, 'profile.html', {'error_message': 'Invalid file type. Please upload an image.'})

        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        uploaded_image_url = fs.url(filename)

        # Path to the uploaded image
        image_path = os.path.join(settings.MEDIA_ROOT, filename)
        img = cv2.imread(image_path)

        if img is None:
            return render(request, 'profile.html', {'error_message': 'Could not read the uploaded image file.'})

        try:
            # Resize the original uploaded image to a consistent size (183x275)
            img_resized_uploaded = cv2.resize(img, (275,183))

            # Save the resized uploaded image for display
            resized_filename = f'resized_{filename}'
            resized_path = os.path.join(settings.MEDIA_ROOT, resized_filename)
            cv2.imwrite(resized_path, img_resized_uploaded)
            resized_image_url = fs.url(resized_filename)

            # Process the resized image for defect detection
            # Convert to grayscale and apply Gaussian blur for noise reduction
            gray = cv2.cvtColor(img_resized_uploaded, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Use adaptive thresholding to better handle varying lighting conditions
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Apply dilation to close gaps between nearby contours and erosion to remove noise
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)

            # Find contours of defects
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_img = img_resized_uploaded.copy()

            detected_areas = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # No filtering on contour size, to ensure even small defects are detected
                if w > 5 and h > 5:  # Lower the size threshold to catch smaller defects
                    detected_areas.append((x, y, w, h))
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(result_img, f"W:{w}, H:{h}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Resize the processed image to the same size as the original uploaded image
            img_resized_processed = cv2.resize(result_img, (275,183))

            # Save the processed image
            processed_filename = f'processed_{filename}'
            processed_path = os.path.join(settings.MEDIA_ROOT, processed_filename)
            cv2.imwrite(processed_path, img_resized_processed)
            processed_image_url = fs.url(processed_filename)

            # Prepare output data with detected areas (defects)
            areas_info = [f"Width={w}px, Height={h}px" for (x, y, w, h) in detected_areas]

            return render(request, 'profile.html', {
                'uploaded_image_url': resized_image_url,  # Display resized image
                'processed_image_url': processed_image_url,  # Display processed image
                'areas_info': areas_info
            })

        except Exception as e:
            return render(request, 'profile.html', {'error_message': f"Error processing the image: {e}"})

    return render(request, 'profile.html')
