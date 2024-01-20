import torch
from sklearn.preprocessing import StandardScaler

"""The input for SaTNC model"""

s_elec_dict = {'Ni': 2,'Al':2,'Co':2,'Cr':1,'Mo':1,'Re':2,'Ru':1,'Ti':2,
               'Ta':2,'W':2,'Hf':2,'Nb':1,'Si':2,'C':2 ,'Y':2,'Ce':2, 'B': 2}

d_elec_dict = {'Ni': 8,'Al':0,'Co':7,'Cr':5,'Mo':5,'Re':5,'Ru':7,'Ti':2,
               'Ta':3,'W':4,'Hf':2,'Nb':4,'Si':0,'C':0 ,'Y':1,'Ce':1, 'B': 0}

f_elec_dict = {'Ni': 0,'Al':0,'Co':0,'Cr':0,'Mo':0,'Re':14,'Ru':0,'Ti':0,
               'Ta':14,'W':14,'Hf':14,'Nb':0,'Si':0,'C':0, 'Y':0,'Ce':1, 'B': 0}

radius_dict = {'Ni': 0.135,'Al':0.125,'Co':0.135,'Cr':0.140,'Mo':0.145,'Re':0.135,'Ru':0.130,'Ti':0.14, 
               'Ta':0.145,'W':0.135,'Hf':0.155,'Nb':0.145,'Si':0.11,'C': 0.07, 'Y':0.18,'Ce':0.185, 'B': 0.085}

atom_mass = {'Ni':58.69,'Al':26.98,'Co':58.93,'Cr':52,'Mo':95.95,'Re':186.2,'Ru':101.07, 'Ti':47.87,
             'Ta':180.94,'W':183.84,'Hf':178.49,'Nb':92.90,'Si':28.085, 'C':12,'Y':88.90,'Ce':140.12,'B':10.81}

Q_calc_dict = {'Ni':(287000, 69.8), 'Al':(284000,59.8), 'Co':(284169,67.6), 'Cr':(287000,64.4), 'Mo':(267585,79.5), 
               'Re':(278817,105), 'Ru':(304489.39,68.97), 'Ti':(256900,77.4), 'Ta':(267729,79.9), 'W':(282130,87.2), 
               'Hf':(251956,71.2), 'Nb':(253446,0), 'Si':(0,0), 'C':(0,0), 'Y':(0,0), 'Ce':(0,0), 'B':(0,0)}

Q_Ni3Al_dict = {'Ni':303, 'Al':258, 'Co':325, 'Cr':366, 'Mo':493, 'Re':467.5, 'Ru':318.7, 'Ti':468, 
                'Ta':425, 'W':0, 'Hf':0, 'Nb':0, 'Si':0, 'C':0, 'Y':0, 'Ce':0, 'B':0}

class Transform:
    def __init__(self, tempreture, at_value, length, 
                    gama_value, gamapie_value, condition_data):
        self.at_value = at_value
        self.gama_value = gama_value
        self.gamapie_value = gamapie_value
        self.condition_data = condition_data
        self.tempreture = tempreture
        self.length = length

    def scaler(self, feature):
        return feature / max([abs(i) for i in feature])

    def Q_calc(self, Temperature):
        Q_dict = {}
        for atom in Q_calc_dict.keys(): 
            a = Q_calc_dict[atom][0]
            b = Q_calc_dict[atom][1]
            Q_dict[atom] = (a + b*Temperature) / 1000
        return Q_dict
    
    def c_embedding(self):
        c = torch.zeros(self.length, 17, 6)
        for i in range(self.length):
            c[i, :, 0: 4] = self.at_value[i].t().reshape(-1, 1).repeat(1, 4)
            c[i, :, 4] = self.gama_value[i].t()
            c[i, :, 5] = self.gamapie_value[i].t()
        return c

    def p_embedding(self):
        p = torch.zeros(self.length, 17, 6)
        for i in range(self.length):
            p[i, :, 0] = torch.from_numpy(self.scaler(s_elec_dict.values())).type(torch.float).t()
            p[i, :, 1] = torch.from_numpy(self.scaler(d_elec_dict.values())).type(torch.float).t()
            p[i, :, 2] = torch.from_numpy(self.scaler(f_elec_dict.values())).type(torch.float).t()
            p[i, :, 3] = torch.from_numpy(self.scaler(radius_dict.values())).type(torch.float).t()
            p[i, :, 4] = torch.from_numpy(self.scaler(self.Q_calc(self.tempreture[i]).values())).type(torch.float).t()
            p[i, :, 5] = torch.from_numpy(self.scaler(Q_Ni3Al_dict.values())).type(torch.float).t()
        return p
    
    def fit(self):
        attention_data = torch.zeros(self.length, 2, 17, 6)
        attention_data[:, 0, :, :] = self.c_embedding()
        attention_data[:, 1, :, :] = self.p_embedding()

        scaler = StandardScaler()
        scaler_values = scaler.fit_transform(self.condition_data)
        scaler_values = torch.from_numpy(scaler_values).type(torch.float)
        return [attention_data, scaler_values]