(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        blue - block
        red - block
        purple - block
        orange - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on blue yellow)
        (on red blue)
        (on purple red)
        (on orange purple)
        (on green orange)
        (clear green)
        ; yellow -> blue -> red -> purple -> orange -> green
    )
    (:goal (and (on green purple)
        (on blue green)
        (on yellow blue)
        (on orange yellow)
        (on red orange)))
)
    