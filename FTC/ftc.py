# Autor: Rodrigo Gael Guzmán Alburo
# Fecha: 30/03/2025


def FTC(database_D, minsup):
    """
    Implementación del algoritmo Frequent Termset Cover (FTC)

    Args:
    database_D: Lista de documentos, donde cada documento es un conjunto de términos
    minsup: Soporte mínimo (valor float entre 0 y 1)

    Returns:
    SelectedTermSets: Lista de conjuntos de términos seleccionados
    cover: Diccionario que mapea cada conjunto de términos a los documentos que cubre
    """

    SelectedTermSets = []
    n = len(database_D)
    RemainingTermSets = DetermineFrequentTermsets(database_D, minsup)

    # Función para calcular la cobertura actual
    def coverage(term_sets):
        cov = set()
        for ts in term_sets:
            cov.update(cov_ts.get(ts, set()))
        return cov

    # Precalculamos la cobertura inicial de todos los term sets
    cov_ts = {ts: set() for ts in RemainingTermSets}
    for i, doc in enumerate(database_D):
        for ts in RemainingTermSets:
            if ts.issubset(doc):
                cov_ts[ts].add(i)

    # Mientras no se cubran todos los documentos
    while len(coverage(SelectedTermSets)) < n:
        min_overlap = float("inf")
        BestCandidate = None

        # Encontrar el mejor candidato con mínimo solapamiento
        for ts in RemainingTermSets:
            # Calcular solapamiento con los ya seleccionados
            current_cover = coverage(SelectedTermSets)
            overlap = len(current_cover.intersection(cov_ts[ts]))

            if overlap < min_overlap or (
                overlap == min_overlap
                and len(cov_ts[ts]) > len(cov_ts.get(BestCandidate, set()))
            ):
                min_overlap = overlap
                BestCandidate = ts

        if (
            BestCandidate is None
        ):  # No hay más candidatos que cubran documentos no cubiertos
            break

        # Añadir el mejor candidato a los seleccionados
        SelectedTermSets.append(BestCandidate)
        RemainingTermSets.remove(BestCandidate)

        # Eliminar documentos cubiertos por el BestCandidate de la cobertura de los demás
        docs_to_remove = cov_ts[BestCandidate]
        for ts in RemainingTermSets:
            cov_ts[ts] = cov_ts[ts].difference(docs_to_remove)

    # Preparar el resultado
    cover = {tuple(ts): cov_ts[ts] for ts in SelectedTermSets}

    return SelectedTermSets, cover


def DetermineFrequentTermsets(D, minsup):
    """
    Función auxiliar para encontrar todos los conjuntos de términos frecuentes

    Args:
    D: Base de datos de documentos
    minsup: Soporte mínimo (valor float)

    Returns:
    Lista de conjuntos de términos que cumplen con el soporte mínimo
    """
    from itertools import combinations
    from collections import defaultdict

    # Contar frecuencia de términos individuales
    term_counts = defaultdict(int)
    for doc in D:
        for term in doc:
            term_counts[term] += 1

    # Filtrar términos frecuentes
    min_count = minsup * len(D)
    frequent_terms = {term for term, count in term_counts.items() if count >= min_count}

    # Generar todos los posibles conjuntos de términos frecuentes
    frequent_termsets = []
    max_k = 10  # Límite práctico para evitar explosión combinatoria

    for k in range(1, max_k + 1):
        # Generar todas las combinaciones de tamaño k
        for termset in combinations(frequent_terms, k):
            # Verificar si es frecuente
            count = 0
            for doc in D:
                if set(termset).issubset(doc):
                    count += 1
            if count >= min_count:
                frequent_termsets.append(frozenset(termset))

    return frequent_termsets


if __name__ == "__main__":
    # Datos de ejemplo
    documents = [
        {"manzana", "platano", "naranja"},  # Doc 0
        {"manzana", "platano"},  # Doc 1
        {"manzana", "naranja"},  # Doc 2
        {"manzana"},  # Doc 3
        {"platano", "naranja"},  # Doc 4
        {"platano"},  # Doc 5
        {"naranja"},  # Doc 6
        {"lechuga", "tomate"},  # Doc 7
        {"lechuga", "tomate", "pepino"},  # Doc 8
        {"lechuga", "pepino"},  # Doc 9
        {"tomate", "pepino"},  # Doc 10
        {"pan", "leche", "huevos"},  # Doc 11
        {"pan", "leche"},  # Doc 12
        {"pan", "huevos"},  # Doc 13
        {"leche", "huevos"},  # Doc 14
        {"pan"},  # Doc 15
        {"leche"},  # Doc 16
        {"huevos"},  # Doc 17
    ]

    # Ejecutar el algoritmo FTC
    minsup = 0.2  # 20% de soporte mínimo
    selected_termsets, coverage = FTC(documents, minsup)

    # Mostrar resultados
    print("Conjuntos de términos seleccionados:")
    for i, termset in enumerate(selected_termsets):
        # Convertimos el frozenset a un set normal para mostrarlo mejor
        termset_set = set(termset)
        # Buscamos la clave correcta en el diccionario coverage
        for key in coverage:
            if set(key) == termset_set:
                print(
                    f"{i+1}. {termset_set} - Cubre {len(coverage[key])} documentos: {coverage[key]}"
                )
                break

    # Verificar cobertura total
    all_covered = set()
    for termset in coverage:
        all_covered.update(coverage[termset])

    print(f"\nDocumentos cubiertos: {len(all_covered)} de {len(documents)}")
    print(
        f"Documentos no cubiertos: {set(range(len(documents))) - all_covered if len(all_covered) < len(documents) else 'Todos cubiertos'}"
    )
