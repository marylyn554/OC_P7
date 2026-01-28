from bs4 import BeautifulSoup

def make_wcag_friendly(html_path: str, lang: str = "fr"):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    # 1) Langue du document
    if soup.html:
        soup.html["lang"] = lang

    # 2) <title> (souvent absent ou pas parlant)
    if soup.head:
        if not soup.head.title:
            title_tag = soup.new_tag("title")
            title_tag.string = "Rapport Data Drift (Evidently)"
            soup.head.insert(0, title_tag)
        else:
            soup.head.title.string = "Rapport Data Drift (Evidently)"

        # 3) Style minimal pour focus visible + skip link
        style = soup.new_tag("style")
        style.string = """
        .skip-link{
          position:absolute; left:-999px; top:auto; width:1px; height:1px; overflow:hidden;
        }
        .skip-link:focus{
          left:1rem; top:1rem; width:auto; height:auto; padding:.6rem 1rem;
          background:#fff; border:2px solid #000; z-index:9999;
        }
        :focus{ outline: 3px solid #000; outline-offset: 2px; }
        """
        soup.head.append(style)

    # 4) Créer un landmark <main> + skip link
    body = soup.body
    if not body:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(str(soup))
        return

    # Skip link
    skip = soup.new_tag("a", href="#main-content")
    skip["class"] = "skip-link"
    skip.string = "Aller au contenu principal"
    body.insert(0, skip)

    # Wrap contenu dans <main>
    main = soup.new_tag("main", id="main-content")
    main["role"] = "main"

    # Déplacer tout sauf le skip-link dans main
    # (on garde le skip-link en premier)
    for child in list(body.contents)[1:]:
        main.append(child.extract())
    body.append(main)

    # 5) Ajouter un H1 si absent (structure de titres)
    # (Evidently n'a pas toujours un vrai h1)
    if not main.find(["h1"]):
        h1 = soup.new_tag("h1")
        h1.string = "Rapport de Data Drift"
        main.insert(0, h1)

    # 6) Ajouter un bloc "Résumé exécutif" (à personnaliser)
    summary = soup.new_tag("section")
    summary["aria-labelledby"] = "exec-summary-title"
    h2 = soup.new_tag("h2", id="exec-summary-title")
    h2.string = "Résumé exécutif (accessible)"
    summary.append(h2)

    p = soup.new_tag("p")
    p.string = (
        "Ce résumé fournit une alternative textuelle aux graphiques interactifs du rapport. "
        "Il met en avant les principaux constats de dérive afin de répondre aux exigences WCAG."
    )
    summary.append(p)

    # 👉 Ici tu peux injecter des chiffres concrets (top PSI etc.)
    ul = soup.new_tag("ul")
    for item in [
        "Méthode de drift : PSI.",
        "Comparer les distributions entre données de référence et données courantes.",
        "Consulter les sections 'Drift par variable' pour identifier les colonnes les plus affectées."
    ]:
        li = soup.new_tag("li")
        li.string = item
        ul.append(li)
    summary.append(ul)

    main.insert(1, summary)

    # 7) Ajouter des aria-label sur les conteneurs de graphes (souvent des <div>)
    # Plotly/Evidently : on donne au moins un label générique
    for i, div in enumerate(main.find_all("div")):
        # On évite de surcharger tout, on cible ceux qui contiennent des graphes
        if div.get("class") and any("plotly" in c.lower() for c in div.get("class")):
            div["role"] = "img"
            div["aria-label"] = f"Graphique interactif {i+1} (voir le résumé exécutif pour la version texte)."
            div["tabindex"] = "0"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))

make_wcag_friendly('notebooks/artifacts/data_drift_report_wcag.html', lang="fr")