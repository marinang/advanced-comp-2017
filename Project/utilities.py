# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
import matplotlib.pyplot as plt

def ReadData(file="diabetic_data.csv"):
#	return record array from the data
#	return np.genfromtxt(file, dtype= None, names=True, delimiter=',', missing_values='?').view(np.recarray)
	return pd.read_csv(file,delimiter=',',na_values='?')
	

def X_y(dataframe):
	
	X = dataframe.drop('readmitted',axis=1).as_matrix()
	y = dataframe['readmitted'].as_matrix()

	for index,val in np.ndenumerate(y):
		if val == "NO" or val == ">30":
			y[index] = "b" #otherwise
		else:
			y[index] = "r" #readmitted
	
	return X,y


class DataEncoder():
	
	def __init__(self, n_HashMedications=10, n_medical_speciality=10, n_admission_threshold = 500, n_dismission_threshold = 500, cols_medications=None, cols_ohe=None, cols_to_drop=None):
		self.n_HashMedications = n_HashMedications
		self.cols_ohe = cols_ohe
		self.cols_to_drop = cols_to_drop
		self.cols_medications = cols_medications
		self.n_medical_speciality = n_medical_speciality
		self.n_admission_threshold = n_admission_threshold
		self.n_dismission_threshold = n_dismission_threshold
		self.default_cols_ohe = ["race","A1Cresult","max_glu_serum","change","diabetesMed"]
		self.default_cols_medications = ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide","glipizide",
						"glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","examide",
						"citoglipton","insulin","glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone","metformin-rosiglitazone",
						"metformin-pioglitazone"]
		
	def transform1(self,df):
		
		df = self.DropFeatures(df)
		
		print("Dropping Features!")
		
		df = self.OneHotEncoding(df)
	
		print("One Hot Encoding!")
		
		df = self.DiagnosisCombination(df)
		
		print("Combinations of the different diagnosis!")
		
		df = self.AgeEncoding(df)
		
		print("Age Encoding!")
		
		return df
		
	def transform2(self,df):
		
		df = self.AdmissionDischargeTreshold(df)
		
		print("Keep {0} most relevant admission sources and {1} most relevant discharge dispositions, and remove the rest!".format(self.n_admission_threshold,self.n_dismission_threshold))
		
		df = self.MedicalSpecialityEncoding(df)
		
		print("Keep {0} most relevant medical specialities!".format(self.n_medical_speciality))
		
		df = self.MedicationsHashing(df)
		
		print("Hash diabete medications into {0} outputs!".format(self.n_HashMedications))
		
		return df
		
	def transform(self,df):
		
		df = self.transform1(df)

		df = self.transform2(df)
		
		print("Encoding of the data has finished successfully")
		
		return df
	
	def DropFeatures(self,df):
		
		return df.drop(["encounter_id","patient_nbr","weight","payer_code","gender"],axis=1)	
		
	def OneHotEncoding(self,df):
		
		if self.cols_ohe == None:
			cols_ohe = self.default_cols_ohe
		else:
			cols_ohe = self.cols_ohe
			
		#race has 2% missing values	
		if "race" in cols_ohe:
			df_ohe = pd.get_dummies(df,columns=["race"],dummy_na=True)
			cols_ohe.remove("race")
			df_ohe = pd.get_dummies(df_ohe,columns=cols_ohe)
		else:	
			df_ohe = pd.get_dummies(df,columns=cols_ohe)
		
		return df_ohe
		
	def AdmissionDischargeTreshold(self,df):
		
		admission_source_id = df.admission_source_id.value_counts().sort_values()
		admission_source_id = admission_source_id[admission_source_id > self.n_admission_threshold].index.values
		
		discharge_disposition_id = df.discharge_disposition_id.value_counts().sort_values()
		discharge_disposition_id = discharge_disposition_id[discharge_disposition_id > self.n_dismission_threshold].index.values
		
		list_s = [pd.Series(),pd.Series()]
		
		for i in df.index.values:
			for s,(name,list_val) in enumerate(zip(["admission_source_id","discharge_disposition_id"],[admission_source_id,discharge_disposition_id])):
				val = df[name][i]
				if val in list_val:
					s_tmp = pd.Series("{0}_{1}".format(name,val),index=[i])
				else:
					s_tmp = pd.Series("{0}_{1}".format(name,"Other"),index=[i])
				list_s[s] = list_s[s].append(s_tmp)
		
		s_admission = pd.get_dummies(list_s[0])
		s_discharge = pd.get_dummies(list_s[1])
		
		
		df_threshold = df.drop(["admission_source_id","discharge_disposition_id"],axis=1)
		df_threshold = pd.concat([df_threshold,s_admission,s_discharge],axis=1)
		
		return df_threshold			
		
		
	def MedicalSpecialityEncoding(self,df):
		
		
		medical_specialty = df.medical_specialty.value_counts(dropna=False).sort_values()
		med_spec = medical_specialty.index.values[-self.n_medical_speciality:-1]
		
		s = pd.Series()
		
		for i,val in df.medical_specialty.iteritems():
			if not isinstance(val, str):
				s_tmp = pd.Series("medical_speciality_Unknown",index=[i])
			elif val in med_spec:
				s_tmp = pd.Series("medical_speciality_"+val,index=[i])
			else:
				s_tmp = pd.Series("medical_speciality_Other",index=[i])
			s = s.append(s_tmp)
		s = pd.get_dummies(s)

		df_med_spec = df.drop("medical_specialty",axis=1)
		df_med_spec = pd.concat([df_med_spec,s],axis=1)
		
		return df_med_spec
		
	def MedicationsHashing(self,df):
		
		if self.cols_medications == None:
			cols_medications = self.default_cols_medications
		else:
			cols_medications = self.cols_medications
			
		FH = FeatureHasher(self.n_HashMedications)
		
		dict_df = df[cols_medications].to_dict('records')
		dict_transf = FH.transform(dict_df)
		
		df_transf = pd.DataFrame(dict_transf.toarray())
		
		#rename columns
		cols = {n:"medications_hash_{0}".format(n) for n in range(self.n_HashMedications)}
#		df_transf = df_transf.drop('Unnamed: 0',axis=1)
		
		df_med_hash = df.drop(cols_medications,axis=1)
		df_med_hash = pd.concat([df_med_hash,df_transf],axis=1)	
		df_med_hash = df_med_hash.rename(index=str,columns=cols)
	
		return df_med_hash
		
	
	def DiagnosisCombination(self,df):
		
		Circulatory_r = np.arange(390,459+1,1)
		Circulatory_r = np.append(Circulatory_r,785)
		Respiratory_r = np.arange(460,519+1,1)
		Respiratory_r = np.append(Respiratory_r,786)
		Digestive_r = np.arange(520,579+1,1)
		Digestive_r = np.append(Digestive_r,787)
		Injury_r = np.arange(800,999+1,1)
		Musculoskeletal_r = np.arange(710,739+1,1)
		Genitourinary_r = np.arange(580,629+1,1)
		Genitourinary_r = np.append(Genitourinary_r,788)
		Neoplasms_r = np.arange(140,239+1,1)
		
		list_s = [pd.Series(),pd.Series(),pd.Series()]
		
		for i in df.index.values:
			for s,name in enumerate(["diag_1","diag_2","diag_3"]):
				val = str(df[name][i])
		
#		for i,val in df.iteritems():
#			val = str(val)
				if "V" in val or "E" in val or "365" in val:
					s_tmp = pd.Series(name+"_Other",index=[i])
				elif "250" in val:
					s_tmp = pd.Series(name+"_Diabetes",index=[i])
				elif float(val) in Circulatory_r:
					s_tmp = pd.Series(name+"_Circulatory",index=[i])
				elif float(val) in Respiratory_r:
					s_tmp = pd.Series(name+"_Respiratory",index=[i])
				elif float(val) in Digestive_r:
					s_tmp = pd.Series(name+"_Digestive",index=[i])
				elif float(val) in Injury_r:
					s_tmp = pd.Series(name+"_Injury",index=[i])
				elif float(val) in Musculoskeletal_r:
					s_tmp = pd.Series(name+"_Musculoskeletal",index=[i])
				elif float(val) in Genitourinary_r:
					s_tmp = pd.Series(name+"_Genitourinary",index=[i])
				elif float(val) in Neoplasms_r:
					s_tmp = pd.Series(name+"_Neoplasms",index=[i])
				else:
					s_tmp = pd.Series(name+"_Other",index=[i])
				list_s[s] = list_s[s].append(s_tmp)
			
		s1 = pd.get_dummies(list_s[0])
		s2 = pd.get_dummies(list_s[1])
		s3 = pd.get_dummies(list_s[2])
		
		df_diag_comb = df.drop(["diag_1","diag_2","diag_3"],axis=1)
		df_diag_comb = pd.concat([df_diag_comb,s1,s2,s3],axis=1)
		
		return df_diag_comb	
	
	def AgeEncoding(self,df):
		
		age = df.age
		
		s = pd.Series()
		
		for i in df.index.values:
			val = age[i]
			if val == "[0-10)" or val == "[10-20)":
				s_tmp = pd.Series("[0-20)",index=[i])
			elif val == "[20-30)" or val == "[30-40)" or val == "[40-50)" or val == "[50-60)":
				s_tmp = pd.Series("[20-60)",index=[i])
			else:
				s_tmp = pd.Series("[60-100)",index=[i])
			s = s.append(s_tmp)
			
		
		
		s = pd.get_dummies(s)

		df_age = df.drop(["age"],axis=1)
		df_age = pd.concat([df_age,s],axis=1)
		
		return df_age		
		
	
def select_data(dataframe,write=False):
	
	list_to_drop = []
	
	#keep only the first encounter of each patient
	
	patient_nbr = dataframe.patient_nbr.value_counts()
	
	patient_nbr = patient_nbr[patient_nbr > 1]

	
	for index in patient_nbr.index:
		tmp_dat = dataframe[dataframe.patient_nbr == index]
		for index,val in enumerate(tmp_dat.index.values):
			if index == 0:
				continue
			else:
				list_to_drop.append(val)
					
	#remove encounters that resulted in either discharge to a hospice or patient death
	#(Died. All episodes of inpatient care that terminated in death. Patient expired after admission and before leaving the hospital.)
	# => discharge_disposition_id 11,13,14,19,20
	
	tmp_dat = dataframe[(dataframe["discharge_disposition_id"] == 11) | (dataframe["discharge_disposition_id"] == 13) | (dataframe["discharge_disposition_id"] == 14) | (dataframe["discharge_disposition_id"] == 19) | (dataframe["discharge_disposition_id"] == 19)]
	
	for val in tmp_dat.index.values:
		list_to_drop.append(val)
	
	data = dataframe.drop(list_to_drop)
	
	if write:
		data.to_csv('diabetic_data_sel.csv',index=False)
	
	return data
	
def correlations(data, **kwds):
	"""Calculate pairwise correlation between features.
		
	Extra arguments are passed on to DataFrame.corr()
		"""
	# simply call df.corr() to get a table of
	# correlation values if you do not need
	# the fancy plotting
	corrmat = data.corr(**kwds)
	
	fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))
	
	opts = {'cmap': plt.get_cmap("RdBu"),
		'vmin': -1, 'vmax': +1}
	heatmap1 = ax1.pcolor(corrmat, **opts)
	plt.colorbar(heatmap1, ax=ax1)

	ax1.set_title("Correlations")
	
	labels = corrmat.columns.values
	for ax in (ax1,):
		# shift location of ticks to center of the bins
		ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
		ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
		ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
		ax.set_yticklabels(labels, minor=False)
	
	#plt.tight_layout()

	#fig.savefig(file)
	
	