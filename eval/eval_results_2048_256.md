# RAG Retriever Evaluation Results

Generated: 2026-06-16 13:29

## Configuration
- Chunk size : 2048 tokens
- Overlap    : 256 tokens
- Eval set   : 189 questions  (specific=100, topic=9, metadata=60, summary=20)

## Overall Results

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | Latency (ms) |
|--------|-------|-------|--------|-----|-------------|
| BM25 only | 0.032 | 0.058 | 0.063 | 0.044 | 68 ms |
| Dense - Cohere | 0.185 | 0.312 | 0.339 | 0.237 | 557 ms |
| **Dense - Cohere + Rerank** | 0.439 | 0.635 | 0.730 | 0.530 | 2866 ms |
| Hybrid Cohere + BM25 | 0.270 | 0.423 | 0.524 | 0.344 | 538 ms |
| Hybrid Cohere + Rerank | 0.449 | 0.599 | 0.658 | 0.513 | 2147 ms |

## Hit@5 by Question Type

| Config | Specific | Topic | Metadata | Summary |
|--------|----------|-------|----------|---------|
| BM25 only | 0.100 | 0.111 | 0.000 | 0.000 |
| Dense - Cohere | 0.240 | 0.778 | 0.350 | 0.350 |
| Dense - Cohere + Rerank | 0.680 | 0.889 | 0.517 | 0.650 |
| Hybrid Cohere + BM25 | 0.340 | 1.000 | 0.450 | 0.500 |
| Hybrid Cohere + Rerank | 0.626 | 0.778 | 0.525 | 0.600 |

## Winner: Dense - Cohere + Rerank
Best Hit@5: 0.635
Improvement over BM25: +990.9%

## Sample Questions by Type

**Specific:** *¿Cuál es la fecha y hora límite para presentar las solicitudes de aclaración sobre las bases de la licitación?*
**Topic:** *¿Qué licitaciones de construcción y pavimentación hay abiertas en Chihuahua?*
**Metadata:** *¿Cuál es el presupuesto estimado o rango de inversión para la rehabilitación del gimnasio y construcción de baños en la secundaria?*
**Summary:** *¿Cuáles son las especificaciones técnicas exactas de los uniformes operativos (materiales, diseño, tallas, cantidades) y cuáles son los términos de entrega, garantía y penalidades por incumplimiento?*