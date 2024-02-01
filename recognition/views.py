from django.shortcuts import render
from .forms import PredictionForm
# Import your model prediction logic here

def home(request):
    return render(request, 'index.html')

from django.shortcuts import render
from .forms import PredictionForm
# Import your model prediction logic here
# Assuming you have a function `predict_with_model(text, model_id)` that takes the text input and model ID

# Mock-up function to simulate fetching models
# Replace this function with actual database query to fetch models
def get_model_choices():
    # Assuming you have a model that stores these choices, you would fetch them from the database
    # For example, ModelChoice.objects.values_list('id', 'name')
    # Here, we'll just hardcode some choices for demonstration purposes
    return [
        (1, 'Model One'),
        (2, 'Model Two'),
        (3, 'Model Three'),
    ]

def index(request):
    prediction = ""
    # Fetch model choices from the database or service
    model_choices = get_model_choices()
    if request.method == 'POST':
        # Pass the model_choices to the form for proper initialization
        form = PredictionForm(request.POST, model_choices=model_choices)
        if form.is_valid():
            text_input = form.cleaned_data['text_area']
            model_choice = form.cleaned_data['model_choice']
            # Add your model prediction logic here, using the text_input and model_choice
            # This would likely involve calling a separate function or service that performs the prediction
            # For example: prediction = predict_with_model(text_input, model_choice)
    else:
        # Initialize the form with the model choices when the request is not POST
        form = PredictionForm(model_choices=model_choices)
    # Include the form and prediction in the context for rendering
    return render(request, 'recognition/index.html', {'form': form, 'prediction': prediction})

