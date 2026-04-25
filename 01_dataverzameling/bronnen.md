# Bronnen infofiche

**Auteur**: `Jakob De Vreese`  
**Bachelorproef**: Overdraagbare unsupervised anomaliedetectie bij hybride HVAC-systemen  
**Academiejaar**: `2025-2026`  

Dit document geeft meer informatie over de kolommen en het gebruik van het bestand `bronnen.csv` dat gebruikt wordt voor het ophalen van de data voor het trainen van het **anomaliedetectiemodel**.

## Kolommen

- **id**: uniek volgnummer rijen csv
- **func_dp_nr**: Functioneel datapunt nummer, geeft het nummer corresponderend met de geïdentificeerde functionele datapunten, nodig om het overdraagbaar unsupervised anomaliedetectiemodel te trainen en te gebruiken.
- **func_dp_naam**: Functioneel Datapunt naam, de naam van het functioneel datapunt
- **func_systeem**: Functioneel systeem waartoe het functioneel datapunt behoort, zoals warmtepomp, gasketel of collector
- **gbs_naam**: De naam van het datapunt op het gbs (object-naam)
- **object-id_gbs**: De object-id van het datapunt (tijdreeks) om de data op te halen via API
- **opmerking**: eventuele opmerkingen zoals wanneer het over geagregeerde datapunten gaat

### Gebruik

Een functioneel datapunt kan verschillende keren voorkomen in de lijst. Zo kan een ruimtetemperatuur het gemiddelde zijn van verschillende temperatuurvoelers in de te onderzoeken zone. 

