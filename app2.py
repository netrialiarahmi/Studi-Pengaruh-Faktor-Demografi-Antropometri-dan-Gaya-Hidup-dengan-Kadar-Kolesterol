import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the classifier
with open('best_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the StandardScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Dictionary for mapping daerah to provinsi
provinsi = {
    'Aceh': ['Banda Aceh'],
    'Sumatera Utara': ['Ujung Padang','Nias','Soposurung','Paya Pasir','Kabanjahe','Tanah Itam Ulu','Medan', 'Pematangsiantar', 'Pangkalan Brandan', 'Sibolga', 'Lhokseumawe'],
    'Sumatera Barat': ['Cupak','Seirampah','Bonjol','Sungai Penuh','Bukit Tinggi','Padang Panjang', 'Bukittinggi', 'Padang'],
    'Sumatera Selatan': ['Tanjung Agung','Palembang'],
    'Jambi':['Jambi'],
    'Riau': ['Ujung Batu','Pekan Baru','Pekanbaru', 'Rengas Pulau'],
    'KepRi':['Tanjungpinang','Tanjung Pinang','Batam'],
    'Lampung': ['Tanjung Karang','Tanjung Gading','Tri Rahayu','Kota Agung','Metro','Kalidadi','Negara Jaya','Liwa','Bumi Dipasena','Teluk Betung', 'B. Lampung', 'Lampung', 'Bandar Lampung', 'Lampung Selatan'],
    'Bengkulu': ['Penago II','Bengkulu'],
    'Bangka':[ 'Tanjung Pandan','Mentok'],
    'Banten': ['Banten','Pandeglang','Serang', 'Tangerang', 'Tanggerang'],
    'Jawa Barat': ['Kuningan','Majalengka','Bekasi','Ciamis','Indramayu','Bogor', 'Bandung', 'Cirebon', 'Sumedang', 'Sukabumi', 'Cianjur', 'Garut', 'Depok', 'Cimahi', 'Tasikmalaya', 'Purwakarta', 'Subang', 'Karawang', 'Cibadak'],
    'Jawa Tengah': ['Karanganyar','Kab. Semarang','Sukoharjo','Jepara','Banjarnegara','Kendal','Solo','Grobogan','Ngawi','Bumi Ayu','Banyumas','Purbalingga','Surakarta','Brebes','Boyolali','Purworejo', 'Semarang', 'Magelang', 'Tegal', 'Salatiga', 'Klaten', 'Kebumen', 'Temanggung', 'Pati', 'Wonogiri', 'Wonosobo', 'Pemalang', 'Pekalongan', 'Sragen', 'Cilacap', 'Kudus'],
    'Jawa Timur': ['Gresik','Pamekasan','Pasuruan','Lamongan','Lumajang','lamongan','Surabaya', 'Malang', 'Sidoarjo', 'Jember', 'Kediri', 'Tulung Agung', 'Madiun', 'Blora', 'Bojonegoro', 'Bondowoso', 'Nganjuk', 'Trenggalek', 'Ponorogo', 'Magetan'],
    'DKI Jakarta': ['Jakarta Utara','Jakarta', 'Kota Administrasi Jakarta Pusat', 'Kota Administrasi Jakarta Barat', 'Kota Administrasi Jakarta Selatan', 'Kota Administrasi Jakarta Utara'],
    'DI Yogyakarta': ['Gunung Kidul','Srimulyo','Bantul', 'Sleman', 'Gunungkidul', 'Kulon Progo', 'Yogyakarta'],
    'Bali': ['Denpasar'],
    'Nusa Tenggara Barat': ['Mataram'],
    'Nusa Tenggara Timur': ['Quelicai'],
    'Kalimantan Timur': ['Samarinda', 'Balikpapan', 'Balipapan'],
    'Kalimantan Selatan': ['Banjarmasin', 'Barabai'],
    'Kalimantan Barat': ['Singkawang','Pontianak', 'Pemangkat'],
    'Sulawesi Selatan': ['Balang Toa','Ujung Pandang', 'Maros','Tana Toraja','Makassar', 'Ujung Baru', 'Sungguminasa', 'Watampone', 'Sosok'],
    'Sulawesi Utara': ['Manado'],
    'Sulawesi Tenggara':['Raha'],
    'Sulawesi Tengah':['Toli - Toli'],
    'Maluku': ['Ambon'],
    'Papua': ['Jayapura', 'Sentani'],
    'Papua Barat':['Manokwari'],
    'Timor Leste':['Dili'],
}

def cari_provinsi(daerah):
    for prov, daerah_daerah in provinsi.items():
        if daerah.lower() in [x.lower() for x in daerah_daerah]:
            return prov
    return 'Provinsi tidak ditemukan'

# Function to preprocess the input data
numerical_columns = ['Usia', 'Tekanan darah (S)', 'Tekanan darah (D)', 'Tinggi badan (cm)', 'Berat badan (kg)', 'Lingkar perut (cm)', 'Glukosa Puasa (mg/dL)', 'Trigliserida (mg/dL)', 'Fat', 'Visceral Fat', 'Masa Kerja']
all_columns = ['Usia', 'Tekanan darah  (S)', 'Tekanan darah  (D)', 'Tinggi badan (cm)',
       'Berat badan (kg)', 'Lingkar perut (cm)', 'Glukosa Puasa (mg/dL)',
       'Trigliserida (mg/dL)', 'Fat', 'Visceral Fat', 'Masa Kerja',
       'Jenis Kelamin_M', 'IMT (kg/m2)_Normal', 'IMT (kg/m2)_Kegemukan',
       'IMT (kg/m2)_Obesitas', 'Provinsi_Banten', 'Provinsi_Bengkulu',
       'Provinsi_DI Yogyakarta', 'Provinsi_DKI Jakarta', 'Provinsi_Jawa Barat',
       'Provinsi_Jawa Tengah', 'Provinsi_Jawa Timur',
       'Provinsi_Kalimantan Barat', 'Provinsi_Kalimantan Selatan',
       'Provinsi_Kalimantan Timur', 'Provinsi_KepRi', 'Provinsi_Lampung',
       'Provinsi_Maluku', 'Provinsi_Nusa Tenggara Barat',
       'Provinsi_Nusa Tenggara Timur', 'Provinsi_Papua', 'Provinsi_Riau',
       'Provinsi_Sulawesi Selatan', 'Provinsi_Sulawesi Tengah',
       'Provinsi_Sulawesi Tenggara', 'Provinsi_Sulawesi Utara',
       'Provinsi_Sumatera Barat', 'Provinsi_Sumatera Selatan',
       'Provinsi_Sumatera Utara', 'Provinsi_Timor Leste']
def preprocess_input(data):
    # Convert Tempat lahir to provinsi
    data['Tempat lahir'] = data['Tempat lahir'].apply(cari_provinsi)
    
    # Convert Jenis Kelamin to Jenis Kelamin_M
    data['Jenis Kelamin_M'] = data['Jenis Kelamin'].apply(lambda x: 1 if x == 'Laki-laki' else 0)
    data = data.drop(columns=['Jenis Kelamin'])

    # Rename and fill Provinsi columns
    for provinsi_name in provinsi.keys():
        col_name = 'Provinsi_' + provinsi_name
        data[col_name] = data['Tempat lahir'].apply(lambda x: 1 if x == provinsi_name else 0)
    
    data = data.drop(columns=['Tempat lahir'])

    # Drop Provinsi columns with all zeros
    cols_to_drop = [col for col in data.columns if col.startswith('Provinsi') and (data[col] == 0).all()]
    data = data.drop(columns=cols_to_drop)

    # Fill remaining missing columns with 0
    missing_cols = set(all_columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    
    # Reorder columns to match the model's expected input
    data = data[all_columns]

    return data



# Function to predict the class
def predict_class(data):
    prediction = classifier.predict(data)
    return prediction

# Streamlit app
def main():
    st.title('Klasifikasi Kadar Kolesterol')
    st.write('By Iris.tentan')

    # Input form with two columns
    col1, col2 = st.columns(2)

    with col1:
        jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        usia = st.number_input('Usia', min_value=0, max_value=120, value=30)
        tekanan_darah_sistolik = st.number_input('Tekanan darah (Sistolik)', min_value=0, max_value=200, value=120)
        tekanan_darah_diastolik = st.number_input('Tekanan darah (Diastolik)', min_value=0, max_value=100, value=80)
        tinggi_badan = st.number_input('Tinggi badan (cm)', min_value=0.0,max_value=200.0,  value=170.0)
        berat_badan = st.number_input('Berat badan (kg)', min_value=0.0, max_value=150.0, value=70.0)
        imt = st.number_input('IMT (kg/m2)', min_value=0.0, value=25.0)

    with col2:
        lingkar_perut = st.number_input('Lingkar perut (cm)', min_value=0.0, value=90.0)
        glukosa_puasa = st.number_input('Glukosa Puasa (mg/dL)', min_value=0, value=100)
        trigliserida = st.number_input('Trigliserida (mg/dL)', min_value=0, value=150)
        fat = st.number_input('Fat', min_value=0.0, value=25.0 , help='Jumlah total lemak tubuh')
        visceral_fat = st.number_input('Visceral Fat', min_value=0.0, value=10.0, help='Jumlah lemak tubuh yang terletak di sekitar organ dalam tubuh, seperti hati, pankreas, dan usus')
        masa_kerja = st.number_input('Masa Kerja', min_value=0.0, value=5.0, help='Jumlah waktu yang telah dihabiskan seseorang dalam bekerja')
        tempat_lahir = st.text_input('Tempat Lahir', '')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Jenis Kelamin': [jenis_kelamin],
        'Usia': [usia],
        'Tekanan darah (S)': [tekanan_darah_sistolik],
        'Tekanan darah (D)': [tekanan_darah_diastolik],
        'Tinggi badan (cm)': [tinggi_badan],
        'Berat badan (kg)': [berat_badan],
        'IMT (kg/m2)': [imt],
        'Lingkar perut (cm)': [lingkar_perut],
        'Glukosa Puasa (mg/dL)': [glukosa_puasa],
        'Trigliserida (mg/dL)': [trigliserida],
        'Fat': [fat],
        'Visceral Fat': [visceral_fat],
        'Masa Kerja': [masa_kerja],
        'Tempat lahir': [tempat_lahir]
    })

    # Make the prediction only if all inputs are filled
    if st.button('Prediksi', key='predict_button', help='Tekan tombol untuk melakukan prediksi', on_click=None, args=None, kwargs=None):
        if not jenis_kelamin or not usia or not tekanan_darah_sistolik or not tekanan_darah_diastolik or not tinggi_badan or not berat_badan or not imt or not lingkar_perut or not glukosa_puasa or not trigliserida or not fat or not visceral_fat or not masa_kerja or not tempat_lahir:
            st.error('Harap isi semua kolom sebelum melakukan prediksi.')
        else:
            # Preprocess the input data
            input_data_scaled = preprocess_input(input_data)
            prediction = predict_class(input_data_scaled)
            
            # Display the prediction result
            if prediction[0] == 0:
                st.image('Normal.png', caption='Normal')
            else:
                st.image('Tinggi.png', caption='Tinggi')


if __name__ == '__main__':
    main()

