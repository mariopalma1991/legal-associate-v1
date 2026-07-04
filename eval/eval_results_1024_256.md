# RAG Retriever Evaluation Results

Generated: 2026-06-16 10:04

## Configuration
- Chunk size : 1024 tokens
- Overlap    : 256 tokens
- Eval set   : 189 questions  (specific=100, topic=9, metadata=60, summary=20)

## Overall Results

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | Latency (ms) |
|--------|-------|-------|--------|-----|-------------|
| BM25 only | 0.032 | 0.048 | 0.058 | 0.039 | 63 ms |
| Dense - Cohere | 0.233 | 0.402 | 0.503 | 0.318 | 494 ms |
| **Dense - Cohere + Rerank** | 0.439 | 0.656 | 0.741 | 0.528 | 2862 ms |
| Hybrid Cohere + BM25 | 0.280 | 0.429 | 0.519 | 0.355 | 524 ms |
| Hybrid Cohere + Rerank | 0.418 | 0.603 | 0.672 | 0.501 | 1867 ms |

## Hit@5 by Question Type

| Config | Specific | Topic | Metadata | Summary |
|--------|----------|-------|----------|---------|
| BM25 only | 0.080 | 0.111 | 0.000 | 0.000 |
| Dense - Cohere | 0.350 | 1.000 | 0.367 | 0.500 |
| Dense - Cohere + Rerank | 0.690 | 0.889 | 0.567 | 0.650 |
| Hybrid Cohere + BM25 | 0.370 | 1.000 | 0.400 | 0.550 |
| Hybrid Cohere + Rerank | 0.630 | 0.778 | 0.517 | 0.650 |

## Winner: Dense - Cohere + Rerank
Best Hit@5: 0.656
Improvement over BM25: +1277.8%

## Sample Questions by Type

**Specific:** *¿Cuál es la fecha y hora límite para presentar las solicitudes de aclaración sobre las bases de la licitación?*
**Topic:** *¿Qué licitaciones de construcción y pavimentación hay abiertas en Chihuahua?*
**Metadata:** *¿Cuál es el presupuesto estimado o rango de inversión para la rehabilitación del gimnasio y construcción de baños en la secundaria?*
**Summary:** *¿Cuáles son las especificaciones técnicas exactas de los uniformes operativos (materiales, diseño, tallas, cantidades) y cuáles son los términos de entrega, garantía y penalidades por incumplimiento?*