# RAG Retriever Evaluation Results

Generated: 2026-06-16 10:34

## Configuration
- Chunk size : 512 tokens
- Overlap    : 128 tokens
- Eval set   : 189 questions  (specific=100, topic=9, metadata=60, summary=20)

## Overall Results

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | Latency (ms) |
|--------|-------|-------|--------|-----|-------------|
| BM25 only | 0.016 | 0.037 | 0.037 | 0.023 | 63 ms |
| Dense - Cohere | 0.217 | 0.434 | 0.519 | 0.311 | 557 ms |
| **Dense - Cohere + Rerank** | 0.441 | 0.649 | 0.713 | 0.531 | 2204 ms |
| Hybrid Cohere + BM25 | 0.228 | 0.439 | 0.513 | 0.322 | 529 ms |
| Hybrid Cohere + Rerank | 0.397 | 0.571 | 0.630 | 0.469 | 1425 ms |

## Hit@5 by Question Type

| Config | Specific | Topic | Metadata | Summary |
|--------|----------|-------|----------|---------|
| BM25 only | 0.050 | 0.222 | 0.000 | 0.000 |
| Dense - Cohere | 0.370 | 0.889 | 0.467 | 0.450 |
| Dense - Cohere + Rerank | 0.680 | 0.778 | 0.567 | 0.684 |
| Hybrid Cohere + BM25 | 0.380 | 0.889 | 0.467 | 0.450 |
| Hybrid Cohere + Rerank | 0.590 | 0.889 | 0.517 | 0.500 |

## Winner: Dense - Cohere + Rerank
Best Hit@5: 0.649
Improvement over BM25: +1652.1%

## Sample Questions by Type

**Specific:** *¿Cuál es la fecha y hora límite para presentar las solicitudes de aclaración sobre las bases de la licitación?*
**Topic:** *¿Qué licitaciones de construcción y pavimentación hay abiertas en Chihuahua?*
**Metadata:** *¿Cuál es el presupuesto estimado o rango de inversión para la rehabilitación del gimnasio y construcción de baños en la secundaria?*
**Summary:** *¿Cuáles son las especificaciones técnicas exactas de los uniformes operativos (materiales, diseño, tallas, cantidades) y cuáles son los términos de entrega, garantía y penalidades por incumplimiento?*