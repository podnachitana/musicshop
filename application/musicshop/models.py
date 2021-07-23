from django.db import models

class MediaType(models.Model):
    """Медианоситель"""

    name = models.CharField(max_length=100, verbose_name='Название мединосителя')

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Медианоситель'
        verbose_name_plural = 'Медианосители'


class Member(models.Model):
    """Музыкант"""

    name = models.CharField(max_length=255, verbose_name='Имя музыканта')
    slug = models.SlugField()

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Музыкант'
        verbose_name_plural = 'Музыканты'


class Genre(models.Model):
    """Музыкальный жанр"""

