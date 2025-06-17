# Puutarhamyymälän asiakasmääräennuste 🌱

Tämä Streamlit-sovellus ennustaa puutarhamyymälän asiakasmäärän seuraaville päiville sääennusteen perusteella. Ennuste pohjautuu lämpötilaan, sateeseen ja viikonpäivään.

## 🔍 Ominaisuudet

- Hakee 7 päivän sääennusteen Kangasalan alueelle (OpenWeatherMap API:n avulla)
- Ennustaa asiakasmäärät yksinkertaisella koneoppimismallilla (lineaarinen regressio)
- Visualisoi lämpötilan ja sateen graafisesti
- Näyttää arvioidut asiakasmäärät taulukkona

## 🧪 Demo

![Kuvakaappaus](kuva_tahan.png) <!-- Lisää halutessasi kuvakaappaus -->

## ⚙️ Asennusohjeet

### 1. Kloonaa tai lataa tämä repositorio

```bash
git clone https://github.com/YOUR_USERNAME/puutarhamyymala-ennuste.git
cd puutarhamyymala-ennuste
