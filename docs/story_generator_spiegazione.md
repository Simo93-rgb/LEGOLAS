# Generazione Storia Narrativa - Flusso Sintetico

1. **Ordina** gli eventi per timestamp cronologico
2. **Raggruppa** eventi con lo stesso timestamp (simultanei)
3. **Calcola** delta temporale in secondi rispetto al gruppo precedente
4. **Traduce** ogni attività dall'italiano all'inglese (usando `translation_cache.json`)
5. **Categorizza** l'attività e applica il template appropriato
6. **Formatta** in paragrafi discorsivi:
   - Evento singolo: "After X seconds, [activity] was performed"
   - Eventi multipli: "After X seconds, the following examinations were performed simultaneously: [activity1] and [activity2]"
7. **Combina** introduzione paziente + paragrafi con doppio a capo
8. **Ritorna** il testo narrativo completo

---

**In pratica**: traccia eventi → ordinamento temporale → raggruppamento → traduzione → formattazione in paragrafi inglesi → testo finale
