
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "Vgcj9ebtmxtr2H8MFRWXi1RhYP5i0cMG9zABZtFxZB0H"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"input_data": [{"field": [['vehicleType', 'yearOfRegistration','gearbox','powerPS',
                               'model','kilometer','monthOfRegistration','fuelType','brand',
                               'notRepairedDamage']], "values": [[0,2002,0,109,1,125000,1,0,0,0]]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/66c5b24d-5a1a-4066-a2e1-7dd787dbf064/predictions?version=2022-11-16', json=payload_scoring,
 headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json()['predictions'][0]['values'][0][0])





