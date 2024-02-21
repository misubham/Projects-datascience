from django.shortcuts import render

def main(request):
    if request.method != 'POST':
        return render(request, 'form.html')
    bmi = request.POST.get('bmi')
    injury = request.POST.get('injury')
    core_strength = request.POST.get('core_strength')
    flexibility = request.POST.get('flexibility')
    
    # psyc = request.POST.get('psyc','')

    personas=[{'ID': '1','BMI': 'Normal','Injury': 'None','Core strength': 'Normal','Flexibility': 'Normal','Persona': 'Persona 1'},
 {'ID': '2',
  'BMI': 'Obese',
  'Injury': 'None',
  'Core strength': 'Normal',
  'Flexibility': 'Normal',
  'Persona': 'Persona 2'},
 {'ID': '3',
  'BMI': 'Normal',
  'Injury': 'Yes',
  'Core strength': 'Normal',
  'Flexibility': 'Normal',
  'Persona': 'Persona 3'},
 {'ID': '4',
  'BMI': 'Normal',
  'Injury': 'None',
  'Core strength': 'Low',
  'Flexibility': 'Normal',
  'Persona': 'Persona 4'},
 {'ID': '5',
  'BMI': 'Normal',
  'Injury': 'None',
  'Core strength': 'Normal',
  'Flexibility': 'Low',
  'Persona': 'Persona 5'},
 {'ID': '6',
  'BMI': 'Obese',
  'Injury': 'Yes',
  'Core strength': 'Normal',
  'Flexibility': 'Normal',
  'Persona': 'Persona 6'},
 {'ID': '7',
  'BMI': 'Obese',
  'Injury': 'Yes',
  'Core strength': 'Normal',
  'Flexibility': 'Normal',
  'Persona': 'Persona 7'},
 {'ID': '8',
  'BMI': 'Obese',
  'Injury': 'None',
  'Core strength': 'Low',
  'Flexibility': 'Normal',
  'Persona': 'Persona 8'},
 {'ID': '9',
  'BMI': 'Obese',
  'Injury': 'None',
  'Core strength': 'Normal',
  'Flexibility': 'Low',
  'Persona': 'Persona 9'},
 {'ID': '10',
  'BMI': 'Obese',
  'Injury': 'Yes',
  'Core strength': 'Low',
  'Flexibility': 'Low',
  'Persona': 'Persona 10'},
 {'ID': '11',
  'BMI': 'Normal',
  'Injury': 'None',
  'Core strength': 'Normal',
  'Flexibility': 'Low',
  'Persona': 'Persona 11'}]
    
    # Initialize best match
    best_match = None
    best_score = 0

# Loop through personas to find best match
    for persona in personas:
        score = 0
        
        if persona['BMI'] == bmi:
            score += 1
        if persona['Injury'] == injury:  
            score += 1
        if persona['Core strength'] == core_strength:
            score += 1
        if persona['Flexibility'] == flexibility:
            score += 1  
            
        if score > best_score:
            best_match = persona
            best_score = score
        
    if best_match:
        # print("Best match is:", best_match['Persona'])
        return render(request, 'result.html', {'persona': best_match})
    else:
        # print("No match found")
        return render(request, 'result.html', {'message': "No match found"})
    
    
    conditions_to_persona = {
        ('normal', 'none', 'normal', 'normal'): "Please refer Persona 1.",
        ('obese', 'none', 'normal', 'normal'): "Please refer Persona 2.",
        ('normal', 'yes', 'normal', 'normal'): "Please refer Persona 3.",
        ('normal', 'none', 'low', 'normal'): "Please refer Persona 4.",
        ('normal', 'none', 'normal', 'low'): "Please refer Persona 5 or 11 .",
        ('obese', 'yes', 'normal', 'normal'): "Please refer Persona 6 or 7.",
        ('obese', 'none', 'low', 'normal'): "Please refer Persona 8.",
        ('obese', 'none', 'normal', 'low'): "Please refer Persona 9.",
        ('obese', 'yes', 'low', 'low'): "Please refer Persona 10.",
    }

    if persona := conditions_to_persona.get((bmi, injury, core_strength, flexibility)):
        return render(request, 'result.html', {'persona': persona})

    if psyc == 'yes':
        return render(request, 'result.html', {'message':"persona 7"})
    elif psyc=='no':
        return render(request, 'result.html', {'message':"persona 6"})     
    else:
        return render(request, 'result.html', {'message': "May you have to contact doctor/keeps you Healthy."})
