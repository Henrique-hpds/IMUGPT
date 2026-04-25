> O README deve conter uma primeira seção abstract, sendo ela um cabeçalho de nível 1 (# Nome do projeto)
# Framework for generating synthetic IMU based on Human Poses

> Descrição do projeto, objetivos e intenções de uso
> Deve conter ao final a Badge (imagem horizontal) do H.IAAC e das Metas associadas ao projeto

This project was developed as part of the Cognitive Architectures research line from 
the Hub for Artificial Intelligence and Cognitive Architectures (H.IAAC) of the State University of Campinas (UNICAMP).
See more projects from the group [here](https://h-iaac.github.io/HIAAC-Index).

<!--Badges-->
[![](https://img.shields.io/badge/-H.IAAC-eb901a?style=for-the-badge&labelColor=black)](https://hiaac.unicamp.br/)

<!--Meta 1: Arquiteturas Cognitivas-->
[![](https://img.shields.io/badge/-Arq.Cog-black?style=for-the-badge&labelColor=white&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4gPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1Ni4wMDQiIGhlaWdodD0iNTYiIHZpZXdCb3g9IjAgMCA1Ni4wMDQgNTYiPjxwYXRoIGlkPSJhcnFjb2ctMiIgZD0iTTk1NS43NzQsMjc0LjJhNi41Nyw2LjU3LDAsMCwxLTYuNTItNmwtLjA5MS0xLjE0NS04LjEtMi41LS42ODksMS4xMjNhNi41NCw2LjU0LDAsMCwxLTExLjEzNi4wMjEsNi41Niw2LjU2LDAsMCwxLDEuMzY4LTguNDQxbC44LS42NjUtMi4xNS05LjQ5MS0xLjIxNy0uMTJhNi42NTUsNi42NTUsMCwwLDEtMi41OS0uODIyLDYuNTI4LDYuNTI4LDAsMCwxLTIuNDQzLTguOSw2LjU1Niw2LjU1NiwwLDAsMSw1LjctMy4zLDYuNDU2LDYuNDU2LDAsMCwxLDIuNDU4LjQ4M2wxLC40MSw2Ljg2Ny02LjM2Ni0uNDg4LTEuMTA3YTYuNTMsNi41MywwLDAsMSw1Ljk3OC05LjE3Niw2LjU3NSw2LjU3NSwwLDAsMSw2LjUxOCw2LjAxNmwuMDkyLDEuMTQ1LDguMDg3LDIuNS42ODktMS4xMjJhNi41MzUsNi41MzUsMCwxLDEsOS4yODksOC43ODZsLS45NDcuNjUyLDIuMDk1LDkuMjE4LDEuMzQzLjAxM2E2LjUwNyw2LjUwNywwLDAsMSw1LjYwOSw5LjcyMSw2LjU2MSw2LjU2MSwwLDAsMS01LjcsMy4zMWgwYTYuNCw2LjQsMCwwLDEtMi45ODctLjczMmwtMS4wNjEtLjU1LTYuNjgsNi4xOTIuNjM0LDEuMTU5YTYuNTM1LDYuNTM1LDAsMCwxLTUuNzI1LDkuNjkxWm0wLTExLjQ2MWE0Ljk1LDQuOTUsMCwxLDAsNC45NTIsNC45NUE0Ljk1Nyw0Ljk1NywwLDAsMCw5NTUuNzc0LDI2Mi43MzlaTTkzNC44LDI1Ny4zMjVhNC45NTIsNC45NTIsMCwxLDAsNC4yMjEsMi4zNDVBNC45Myw0LjkzLDAsMCwwLDkzNC44LDI1Ny4zMjVabS0uMDIyLTEuNThhNi41MTQsNi41MTQsMCwwLDEsNi41NDksNi4xTDk0MS40LDI2M2w4LjA2MSwyLjUuNjg0LTEuMTQ1YTYuNTkxLDYuNTkxLDAsMCwxLDUuNjI0LTMuMjA2LDYuNDQ4LDYuNDQ4LDAsMCwxLDIuODQ0LjY1bDEuMDQ5LjUxOSw2LjczNC02LjI1MS0uNTkzLTEuMTQ1YTYuNTI1LDYuNTI1LDAsMCwxLC4xMTUtNi4yMjksNi42MTgsNi42MTgsMCwwLDEsMS45NjYtMi4xMzRsLjk0NC0uNjUyLTIuMDkzLTkuMjIyLTEuMzM2LS4wMThhNi41MjEsNi41MjEsMCwwLDEtNi40MjktNi4xbC0uMDc3LTEuMTY1LTguMDc0LTIuNS0uNjg0LDEuMTQ4YTYuNTM0LDYuNTM0LDAsMCwxLTguOTY2LDIuMjY0bC0xLjA5MS0uNjUyLTYuNjE3LDYuMTMxLjc1MSwxLjE5MmE2LjUxOCw2LjUxOCwwLDAsMS0yLjMsOS4xNjRsLTEuMS42MTksMi4wNiw5LjA4NywxLjQ1MS0uMUM5MzQuNDc1LDI1NS43NSw5MzQuNjI2LDI1NS43NDQsOTM0Ljc3OSwyNTUuNzQ0Wm0zNi44NDQtOC43NjJhNC45NzcsNC45NzcsMCwwLDAtNC4zMTYsMi41LDQuODg5LDQuODg5LDAsMCwwLS40NjQsMy43NjIsNC45NDgsNC45NDgsMCwxLDAsNC43NzktNi4yNjZaTTkyOC43LDIzNS41MzNhNC45NzksNC45NzksMCwwLDAtNC4zMTcsMi41LDQuOTQ4LDQuOTQ4LDAsMCwwLDQuMjkxLDcuMzkxLDQuOTc1LDQuOTc1LDAsMCwwLDQuMzE2LTIuNSw0Ljg4Miw0Ljg4MiwwLDAsMCwuNDY0LTMuNzYxLDQuOTQsNC45NCwwLDAsMC00Ljc1NC0zLjYzWm0zNi43NzYtMTAuMzQ2YTQuOTUsNC45NSwwLDEsMCw0LjIyMiwyLjM0NUE0LjkyMyw0LjkyMywwLDAsMCw5NjUuNDc5LDIyNS4xODdabS0yMC45NTItNS40MTVhNC45NTEsNC45NTEsMCwxLDAsNC45NTEsNC45NTFBNC45NTcsNC45NTcsMCwwLDAsOTQ0LjUyNywyMTkuNzcyWiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTkyMi4xNDMgLTIxOC4yKSIgZmlsbD0iIzgzMDNmZiI+PC9wYXRoPjwvc3ZnPiA=)](https://h-iaac.github.io/HIAAC-Index)

> As seguintes seções são obrigatórias, devendo começar com um cabeçalho de nível 2 (## Nome da seção), mantendo o nome delas. No caso de nomes condicionais (A / B), é possível manter assim ou escolher apenas um nome (## A / B ou ## A ou ## B)
>
> Caso deseje, pode adicionar outras seções além das obrigatórias (utilize apenas cabeçalhos de nível 2 e maior)

## Repository Structure
> Lista e descrição das pastas e arquivos importantes na raiz do repositório

- \<pasta>: \<descrição>


## Dependencies / Requirements

> Descrição do passo-a-passo para instalação de bibliotecas, softwares e demais ferramentas
> ncessárias para execução do projeto antes de se clonar o repositório, assim como possíveis
> requisitos mínimos para o projeto (processador, gpu, compilador, etc).

## Installation / Usage

> Passo-a-passo para execução do projeto localmente, assim como parâmetros de configuração
> aceitos (por exemplo, como trocar o caminho para o arquivo de entrada ou saída). No caso de 
> bibliotecas/API fornecer o link para a documentação do mesmo se disponível.

## Citation

> Forneça o Bibtex para citação do repositório.
>
> Ele deve ser um bloco de código do tipo bibtex (\```bibtex CITAÇÃO \```), contendo uma citação do tipo ```@software```, para o repositório. Existe um script para gerar a citação automaticamente (veja ao final deste arquivo).
>
> A primeira citação deve ser ao código do repositório. Citação a outras produções relacionadas podem vir em seguida.

<!--Don't remove the following tags, it's used for placing the generated citation from the CFF file-->
<!--CITATION START-->
```bibtex
@software{

}
```
<!--CITATION END-->

## Authors

> Lista ordenada por data das pessoas que desenvolveram algo neste repositório. Deve ter ao menos 1 autor. Inclua todas as pessoas envolvidas no projeto e desenvolvimento
>
> Você também pode indicar um link para o perfil de algum autor: \[Nome do Autor]\(Link para o Perfil)
  
- (\<ano início>-\<ano fim>) \<Nome>: \<degree>, \<instituição>
  
## Acknowledgements

> Agradecimento as intituições de formento.

>Outros arquivos e informações que o repositório precisa ter:
> - Preencha a descrição do repositóio
>   - É necessário um _role_ de _admin_ no repositório para alterar sua descrição. Pessoas com [_role_ de _owner_](https://github.com/orgs/H-IAAC/people?query=role%3Aowner) na organização do GitHub podem alterar os papéis por repositório.
>   - Na página principal do repositório, na coluna direita, clique na engrenagem ao lado de "About"
>   - É recomendável também adicionar "topics" aos dados do repositório
> - Um arquivo LICENSE contendo a licença do repositório. Recomendamos a licença [LGPLv3](https://choosealicense.com/licenses/lgpl-3.0/).
>   - Converse com seu orientador caso acredite que essa licença não seja adequada. 
> - Um arquivo CFF contendo as informações sobre como citar o repositório.
>   - Este arquivo é lido automaticamente por ferramentas como o próprio GitHub ou o Zenodo, que geram automaticamente as citações.
>   - Existem ferramentas para auxiliar a criação do arquivo, como o [CFF Init](https://bit.ly/cffinit).    
>   - O script `generate_citation.py` pode ser utilizado para preencher o bloco de citação deste README automaticamente:
>     - ```bash
>         python -m pip install cffconvert
>         python generate_citation.py
>         ```
>   - Caso o arquivo tenha a tag `doi: <DOI>`, ele será lido automaticamente pelo Index.
> - Opcionalmente, o repositório pode ser preservado utilizando o Zenodo, que gerará um DOI para ele. [Tutorial](https://help.zenodo.org/docs/github/enable-repository/).
>   - É necessário um _role_ de _admin_ no repositório para publicar um repositório utilizando o Zenodo. Pessoas com [_role_ de _owner_](https://github.com/orgs/H-IAAC/people?query=role%3Aowner) na organização do GitHub podem alterar os papéis por repositório.