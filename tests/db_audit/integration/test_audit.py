# async def main() -> None:
#     WORKITEM_ELEMENT_ID = "Test"
#     PATHS = [
#         "./data/Test/Test.pdf",
#         "./data/Test/Test.docx",
#         "./data/Test/Test.csv",
#         "./data/Test/Test.png",
#         "./data/Test/Test.xlsx"
        
#     ]
#     DOC_IDS = [
#         "./data/Test/Test.pdf",
#         "./data/Test/Test.docx",
#         "./data/Test/Test.csv",
#         "./data/Test/Test.png",
#         "./data/Test/Test.xlsx"
#     ]
#     REQUIREMENTS = [
#         "Zugriffsberechtigungen von Benutzern unter Verantwortung des Unternehmens (interne und externe Mitarbeiter) werden mindestens jährlich überprüft, um diese zeitnah auf Änderungen im Beschäftigungsverhältnis (Kündigung, Versetzung, längerer Abwesenheit/Sabbatical/Elternzeit) anzupassen.",
#         "Die Überprüfung erfolgt durch hierzu autorisierte Personen aus den Unternehmensbereichen des Unternehmens, die aufgrund ihres Wissens über die Zuständigkeiten die Angemessenheit der vergebenen Berechtigungen überprüfen können.",
#         "Die Überprüfung sowie die sich daraus ergebenden Berechtigungsanpassungen werden nachvollziehbar dokumentiert.",
#         "Administrative Berechtigungen werden regelmäßig (mind. jährlich) überprüft.",
#     ]

#     await audit(
#         WORKITEM_ELEMENT_ID,
#         PATHS,
#         DOC_IDS,
#         REQUIREMENTS,
#         skip_indexing=True
#     )


# if __name__ == "__main__":
#     asyncio.run(main())