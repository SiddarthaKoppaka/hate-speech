from django import forms

class PredictionForm(forms.Form):
    text_area = forms.CharField(
        widget=forms.Textarea(attrs={'placeholder': 'Enter your text here'}),
        label='Enter your text',
        required=True
    )
    model_choice = forms.ChoiceField(choices=[], required=True)

    def __init__(self, *args, **kwargs):
        model_choices = kwargs.pop('model_choices', [])
        super(PredictionForm, self).__init__(*args, **kwargs)
        self.fields['model_choice'].choices = model_choices
