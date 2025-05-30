import pandas as pd
data=pd.read_csv('IOTmalw.csv')
#print(data.conn_state.value_counts())
print(data.history.value_counts())
#print(data.Target.value_counts())
#print(data.proto.value_counts())


import pandas as pd

# Assuming Gagfyt is your DataFrame and it is already loaded
# Define the mapping

#protocol_mapping = {'udp': 1, 'tcp': 0, 'icmp': 2}
#conn_state_mapping={'OTH':1, 'SF':0,'S0':2,'OTH':3,'REJ':4}
history_mapping={'S':0,'DdA':1,'Dd':2,'D':3,'DdAFaf':4,'Sr':5,'x':6}
#Target_mapping={'Benign':0,' Malicious':1}


# Apply the mapping to the 'proto' column
#data['proto_encoded'] = data['proto'].map(protocol_mapping)
#data['conn_state_encoded'] = data['conn_state'].map(conn_state_mapping)
data['history_encoded']=data['history'].map(history_mapping)
#data['Target_encoded']=data['Target'].map(Target_mapping)

data.to_csv('IOTmalw.csv',index=False)





# Display the DataFrame to verify the changes
print(data)
