# UAB Sveikata - Mankštos Rekomenduotojas

Streamlit aplikacija, kuri naudoja OpenRouter API ir Qwen3 14B modelį, kad pateiktų personalizuotas mankštos rekomendacijas.
Reikia nemažai padirbėti su stiliumi, bet pavyko padaryti gan greitai ir lengvai, be didesnių problemų, kas labai nudžiugino.

## Funkcionalumas

- **Personalizuotos mankštos programos**: Gauna vartotojo duomenis (amžius, sveikatos problemos, galimas laikas, tikslai) ir sukuria savaitės mankštos planą
- **AI pokalbiai**: Interaktyvus pokalbių interfeisas su AI asistentu sveikatos ir mankštos temomis  
- **Saugus**: AI agentas atsako tik į sveikatos ir mankštos klausimus
- **Lietuvių kalba**: Visas interfeisas ir atsakymai lietuviškai

## Naudojimas

### Reikalavimai
- Python 3.8+
- API raktas iš OpenRouter

### Įdiegimas

1. Klonuokite repozitoriją:
```bash
git clone <repo-url>
cd individualus6
```

2. Įdiekite priklausomybes:
```bash
pip install -r requirements.txt
```

3. Užtikrinkite, kad `api_key_openrouter.txt` faile yra jūsų OpenRouter API raktas

4. Paleiskite aplikaciją:
```bash
streamlit run app.py
```

### Kaip naudoti

1. Atidarykite programą naršyklėje
2. Užpildykite savo duomenis šoninėje juostoje:
   - Amžius
   - Sveikatos problemos
   - Galimas laikas mankštai per dieną  
   - Tikslai (numesti svorio / priaugti raumenų)
3. Spauskite "Gauti mankštos rekomendaciją"
4. Peržiūrėkite sugeneruotą savaitės mankštos programą
5. Užduokite papildomus klausimus pokalbių lange

## Technologijos

- **Streamlit**: Web aplikacijos karkasas
- **OpenRouter API**: AI modelių prieiga
- **Qwen3 14B**: Kalbos modelis rekomendacijoms generuoti
- **OpenAI Python biblioteka**: API sąsaja

## Saugumas

- API raktas saugomas atskirame faile
- AI agentas apribotas tik sveikatos temomis
- Visi atsakymai įspėja, kad tai nėra profesionali medicininė nuomonė
