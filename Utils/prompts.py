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
            ]

    @staticmethod
    def MapReduce_final_prompt(text):
        return [
                    {
                        "role": "user",
                        "content": f"""Vous avez plusieurs rapports détaillés dans l'ordre chronologique d'une transcription.
                                        Votre tâche est d'arrangez de manière cohérente tout les rapports détaillés en un unique document. Veuillez faire ce qui suit :
                                        - Affinez les sujets, y compris leurs points clés, leurs implications, leurs contexte et leurs perspectives.
                                        - Organisez les sujets en chapitres distincts.
                                        - Générez le texte dans un ton professionnel et formel, adapté à un rapport d'entreprise.
                                        - Formatez le texte en Markdown.
                                        Voici les differents rapports détaillés : \n\n{text}\n"""
                    },
                ]

    @staticmethod
    def Refined_recursive_prompt(existing_summary,new_text):
        return [
                {
                    "role": "user",
                    "content": f"""Vous avez un rapport détaillé existant d'une transcription audio et la suite du rapport de la transcription.
                                    Votre tâche est d'affiné le rapport détaillé existant en y incluant la suite du rapport. Veuillez faire ce qui suit :
                                    - Mettre à jour les chapitres dans un ordre chronologique en incluant le nouveau rapport.
                                    - Conserver les points clés, les implications, le contexte du rapport existant.
                                    - Générez le texte dans un ton professionnel et formel, adapté à un rapport d'entreprise.
                                    - Formatez le texte en Markdown.
                                    Voici le rapport existant : \n\n{existing_summary}\n\n
                                    - Affiné le rapport existant pour y ajouter les nouvelles informations ci-dessous:
                                    \n\n{new_text}\n"""
                },
            ]

    @staticmethod
    def Combined_prompt(MapReduce,Refine):
        return  [
                        {
                            "role": "user",
                            "content": f"""Vous avez deux rapports d'une même transcription audio.
                                            Votre tâche est d'affinez un rapport final détaillé à partir de ces deux rapports. Veuillez faire ce qui suit :
                                            - Mettre à jour les chapitres dans un ordre logique.
                                            - Mettez en évidence les aspects uniques de chaque chapitres, y compris leurs points clés.
                                            - Générez le texte dans un ton professionnel et formel, adapté à un rapport d'entreprise.
                                            - Formatez le texte en Markdown.
                                            Voici le premier rapports : \n\n{MapReduce}\n\n
                                            Voici le deuxième rapports : \n\n{Refine}\n"""
                        },

            ]