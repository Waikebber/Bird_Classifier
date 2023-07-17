import os
from tqdm import tqdm
import labelbox as lb
from labelbox import Project, Dataset
from labelbox.exceptions import InvalidAttributeError
class LabelBox:
    def __init__(self, api_key, directory_path=".\\..\\data\\raw\\", project_name="Birds", ontology_name ="Birds"):
        self.api_key = api_key
        self.directory_path = directory_path
        self.ontology =None
        client = lb.Client(api_key=self.api_key)
        self.project = client.create_project(name=project_name,media_type=lb.MediaType.Image,queue_mode='BATCH')
        errors = self.labelbox_proj(ontology_name)
        for error in errors:
            print(f'ERROR UPLOADING DATASET: {error}')
        
    def labelbox_proj(self, ontology_name):
        """Creates LabelBox datasets for every class in the Project.
            Also makes an ontology with each class for the Project.

        Returns:
            lst: List of datasets that had an error being created
        """
        subfolders = [subfolder for subfolder in os.listdir(self.directory_path) if os.path.isdir(os.path.join(self.directory_path, subfolder))]
        for subfolder in tqdm(subfolders):
            self.labelbox_dataset(os.path.join(self.directory_path,subfolder),subfolder)
        errors = self.find_error_sets()
        self.ontology = self.make_ontology(ontology_name)
        return errors
    
    def make_ontology(self,ontology_name):
        """Creates the ontology for the project.
            Each class is a folder name in the init directory.

        Returns:
            labelbox.Ontology: Ontology Object regarding the project
        """
        client = lb.Client(api_key=self.api_key)      
        subfolders = [subfolder for subfolder in os.listdir(self.directory_path) if os.path.isdir(os.path.join(self.directory_path, subfolder))]
        tools = [ lb.Tool(tool=lb.Tool.Type.BBOX, name=name) for name in subfolders ]
        ontology_builder = lb.OntologyBuilder(tools)
        ontology = client.create_ontology(ontology_name,
                                    ontology_builder.asdict())
        self.project.setup_editor(ontology)
        return ontology.uid
    
    def labelbox_dataset(self, directory,dataset_name, retry=0):
        """Creates a dataset in labelbox from all the files in a given dataset. 
            Named after the dataset_name.

        Args:
            directory (str): Directory path to be made into dataset
            dataset_name (str): Name of the dataset to be made
        """
        client = lb.Client(api_key=self.api_key)
        dataset = client.create_dataset(name=dataset_name)
        local_file_paths = [ os.path.join(directory ,x) for x in os.listdir(os.path.join(directory))]
        try:
            task = dataset.create_data_rows(local_file_paths)
            task.wait_till_done()
            self.labelbox_batch(dataset_name, dataset)
        except Exception as err:
            if retry < 3:
                dataset.delete()
                retry +=1
                self.labelbox_dataset(directory, dataset_name, retry)

    
    def labelbox_dataset_lst(self, dataset_name, data_lst):
        """ Add the files in the list of data to the dataset one at a time.
            Dataset is a str name, not actual labelbox Dataset
            Use function for finding bad data. Its slow.

        Args:
            directory (str): Name of dataset to be appended to
            data_lst (lst): lst of file paths to be uploaded

        Raises:
            Exception: When there is an issue uploading a file

        Returns:
            lst: lst of file paths that had an error while uploading
        """     
        client = lb.Client(api_key=self.api_key)
        dataset = None
        try:
            datasets = client.get_datasets(where=(Dataset.name == dataset_name.replace("'", '-')))
            dataset = next(datasets)
        except Exception:
            raise Exception(f"Could not find {dataset_name} in datasets.")
        error_files = []
        for data in data_lst:
            try:
                task = dataset.create_data_rows([data])
                task.wait_till_done()
            except Exception:
                error_files.append(data)
                if not os.path.exists(data):
                    print(f'FileNotFound: {data}')
        self.labelbox_batch(dataset_name,dataset)
        return error_files
                
    def labelbox_batch(self, batch_name, dataset):
        """Creates a batch from an entire dataset.

        Args:
            batch_name (str): Name of the batch
            dataset (labelbox.dataset): dataset to be added to a batch

        Returns:
            labelbox.Batch: A batch object with the new dataset
        """        
        data_row_ids = [dr.uid for dr in dataset.export_data_rows()]
        batch = self.project.create_batch(
            batch_name, # Each batch in a project must have a unique name
            data_row_ids, # A list of data rows or data row ids
            5 # priority between 1(Highest) - 5(lowest)
            )
        return batch
        
    def find_error_sets(self):
        """ Finds all datasets in the project that have no entries. 

        Returns:
            lst: List of all the classes that didn't have entries
        """        
        error_lst = []
        client = lb.Client(api_key=self.api_key)
        try:
            dataset = client.get_datasets()
            for data in dataset:
                if data.row_count == 0:
                    error_lst.append(data.name)
        except InvalidAttributeError:
            print("No Projects connected to Datasets.")
        return error_lst

    def remove_datasets(self):
        """Removes all datasets in the project
        """      
        client = lb.Client(api_key=self.api_key) 
        try: 
            datasets = client.get_datasets()
            for data in datasets:
                data.delete()
        except InvalidAttributeError:
            print("No Projects connected to Datasets.")
    
    def remove_proj(self):
        """Deletes the project from labelbox
        """
        self.project.delete()
    
def remove_all(api_key, ontology_name = ""):
    """Deletes all Projects/Datasets/Ontologies the user can access.
    !!! DO NOT RUN UNLESS YOU ARE SURE YOU WANT TO DELETE EVERYTHING

    Args:
        api_key (str): LabelBox API key
        ontology_name (str, optional): String to be contained in deleted ontologies
    """    
    client = lb.Client(api_key=api_key)
    all_projects = client.get_projects()
    all_datasets = client.get_datasets()
    for proj in all_projects:
        proj.delete()
    
    for data in all_datasets:
        data.delete()
    for ont in client.get_ontologies(ontology_name):
        client.delete_unused_ontology(str(ont.uid))