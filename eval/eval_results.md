# RAG Retriever Evaluation Results

Generated: 2026-06-15 18:20

## Configuration
- Chunk size : 1024 tokens
- Overlap    : 256 tokens
- Eval set   : 189 questions  (specific=100, topic=9, metadata=60, summary=20)

## Overall Results

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | Latency (ms) |
|--------|-------|-------|--------|-----|-------------|
| BM25 only | 0.016 | 0.026 | 0.032 | 0.021 | 139 ms |
| Dense - Cohere | 0.180 | 0.392 | 0.508 | 0.284 | 1251 ms |
| **Dense - Cohere + Rerank** | 0.339 | 0.614 | 0.693 | 0.460 | 2180 ms |
| Hybrid Cohere + BM25 | 0.191 | 0.404 | 0.511 | 0.297 | 968 ms |
| Hybrid Cohere + Rerank | 0.296 | 0.577 | 0.624 | 0.413 | 1414 ms |

## Hit@5 by Question Type

| Config | Specific | Topic | Metadata | Summary |
|--------|----------|-------|----------|---------|
| BM25 only | 0.030 | 0.222 | 0.000 | 0.000 |
| Dense - Cohere | 0.280 | 0.889 | 0.467 | 0.500 |
| Dense - Cohere + Rerank | 0.630 | 0.778 | 0.567 | 0.600 |
| Hybrid Cohere + BM25 | 0.310 | 0.875 | 0.467 | 0.500 |
| Hybrid Cohere + Rerank | 0.580 | 0.889 | 0.550 | 0.500 |

## Winner: Dense - Cohere + Rerank
Best Hit@5: 0.614
Improvement over BM25: +2220.0%

## Sample Questions by Type

**Specific:** *¿Cuál es la fecha y hora límite para presentar las solicitudes de aclaración sobre las bases de la licitación?*
**Topic:** *¿Qué licitaciones de construcción y pavimentación hay abiertas en Chihuahua?*
**Metadata:** *¿Cuál es el presupuesto estimado o rango de inversión para la rehabilitación del gimnasio y construcción de baños en la secundaria?*
**Summary:** *¿Cuáles son las especificaciones técnicas exactas de los uniformes operativos (materiales, diseño, tallas, cantidades) y cuáles son los términos de entrega, garantía y penalidades por incumplimiento?*