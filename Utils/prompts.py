# prompts.py

# Define your prompts

class Prompts:
    @staticmethod
    def summarize_prompt(text):
        return [
            {
                "role": "user",
                "content": f"""Vous avez plusieurs segments d'une transcription audio sur divers sujets.
                                Votre tâche est de créer un rapport détaillé à partir de ces informations. Veuillez faire ce qui suit :
                                - Organisez les sujets en chapitres distincts.
                                - Fournissez une description détaillée pour chaque sujet, y compris ses points clés, ses implications, son contexte et ses perspectives.
                                - Mettez en évidence les aspects uniques de chaque sujet et comment ils contribuent à leur domaine respectif.
                                - Générez le texte dans un ton professionnel et formel, adapté à un rapport d'entreprise.
                                - Formatez le texte en Markdown.
                                Voici les résumés : \n\n{text}\n"""
            },
            {
                "role": "assistant",
                "content": """# {Titre du Sujet}

                                ## Aperçu
                                {Brève description du sujet}

                                ## Points Clés
                                {Description détaillée des points clés du sujet}

                                ## Implications
                                {Explication des implications du sujet}

                                ## Contexte
                                {Description du contexte du sujet}

                                ## Perspectives
                                {Discussion sur les perspectives du sujet}

                                ## Contribution
                                {Explication de la manière dont le sujet contribue à son domaine respectif}"""
            }]

    @staticmethod
    def MapReduce_final_prompt(text):
        return [
                    {
                        "role": "user",
                        "content": f"""Vous avez plusieurs résumés de plusieurs d'une même transcription audio.
                                        Votre tâche est de créer un rapport détaillé à partir de ces résumés. Veuillez faire ce qui suit :
                                        - Organisez les sujets en chapitres distincts.
                                        - Fournissez une description détaillée pour chaque sujet, y compris ses points clés, ses implications, son contexte et ses perspectives.
                                        - Mettez en évidence les aspects uniques de chaque sujet et comment ils contribuent à leur domaine respectif.
                                        - Générez le texte dans un ton professionnel et formel, adapté à un rapport d'entreprise.
                                        - Formatez le texte en Markdown.
                                        Voici les résumés : \n\n{text}\n"""
                    },
                    {
                        "role": "assistant",
                        "content": """# {Titre du Sujet}

                                        ## Aperçu
                                        {Brève description du sujet}

                                        ## Points Clés
                                        {Description détaillée des points clés du sujet}

                                        ## Implications
                                        {Explication des implications du sujet}

                                        ## Contexte
                                        {Description du contexte du sujet}

                                        ## Perspectives
                                        {Discussion sur les perspectives du sujet}

                                        ## Contribution
                                        {Explication de la manière dont le sujet contribue à son domaine respectif}"""
                    }
                ]

    @staticmethod
    def Refined_recursive_prompt(existing_summary,new_text):
        return [
                {
                    "role": "user",
                    "content": f"""Vous avez un résumé existant d'un segment d'une transcription audio et de nouveaux segments de la transcription.
                                    Votre tâche est de créer un rapport détaillé à partir du résumé existant et des nouvelles informations. Veuillez faire ce qui suit :
                                    - Organisez les sujets en chapitres distincts.
                                    - Fournissez une description détaillée pour chaque sujet, y compris ses points clés, ses implications, son contexte et ses perspectives.
                                    - Mettez en évidence les aspects uniques de chaque sujet et comment ils contribuent à leur domaine respectif.
                                    - Générez le texte dans un ton professionnel et formel, adapté à un rapport d'entreprise.
                                    - Formatez le texte en Markdown.
                                    Voici le résumé existant : \n\n{existing_summary}\n\n
                                    Voici les nouvelles informations : \n\n{new_text}\n"""
                },
                {
                    "role": "assistant",
                    "content": """# {Titre du Sujet}

                                    ## Aperçu
                                    {Brève description du sujet}

                                    ## Points Clés
                                    {Description détaillée des points clés du sujet}

                                    ## Implications
                                    {Explication des implications du sujet}

                                    ## Contexte
                                    {Description du contexte du sujet}

                                    ## Perspectives
                                    {Discussion sur les perspectives du sujet}

                                    ## Contribution
                                    {Explication de la manière dont le sujet contribue à son domaine respectif}"""
                }
            ]

    @staticmethod
    def Combined_prompt(MapReduce,Refine):
        return  [
                        {
                            "role": "user",
                            "content": f"""Vous avez deux rapports d'une même transcription audio.
                                            Votre tâche est d'affinez un rapport final détaillé à partir de ces deux rapports. Veuillez faire ce qui suit :
                                            - Réorganisez les chapitres.
                                            - Remettre de l'ordre et de la cohérence.
                                            - Mettez en évidence les aspects uniques de chaque chapitres, y compris ses points clés.
                                            - Générez le texte dans un ton professionnel et formel, adapté à un rapport d'entreprise.
                                            - Formatez le texte en Markdown.
                                            Voici le premier résumés : \n\n{MapReduce}\n\n
                                            Voici le deuxième résumé : \n\n{Refine}\n"""
                        },
                        {
                            "role": "assistant",
                            "content": """# {Titre du Sujet}

                                            ## Aperçu
                                            {Brève description du sujet}

                                            ## Points Clés
                                            {Description détaillée des points clés du sujet}

                                            ## Implications
                                            {Explication des implications du sujet}

                                            ## Contexte
                                            {Description du contexte du sujet}

                                            ## Perspectives
                                            {Discussion sur les perspectives du sujet}

                                            ## Contribution
                                            {Explication de la manière dont le sujet contribue à son domaine respectif}"""
                        }
            ]